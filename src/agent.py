import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import os

load_dotenv()

from src.state import AgentState
from src.tools import validate_schema, compare_scoring_pipelines, score_model, test_model_load

# Initialize the chosen LLM from environment variable, defaulting to llama3.2
model_name = os.environ.get("OLLAMA_MODEL", "llama3.2")
llm = ChatOllama(model=model_name, temperature=0)

# Bind tools
tools = [validate_schema, compare_scoring_pipelines, score_model, test_model_load]
llm_with_tools = llm.bind_tools(tools)

system_prompt = """You are AiDa, an advanced AI diagnostic agent for ML pipelines.
You must follow a STRICT sequential workflow.

CURRENT STATE:
Artifacts uploaded so far: {artifacts}

REQUIRED FILE MAPPINGS:
- If the issue is a "score mismatch", you NEED EXACTLY: `dataset` and `model.json`
- If the issue is a "model load failure", you NEED EXACTLY: `model.json` and `config`

RULES:
1. STEP 1 (Artifact Extraction): If the user describes an issue and `Artifacts uploaded so far` is "None", you CANNOT diagnose. 
   You MUST reply to the user telling them EXACTLY what files they need to upload based on the REQUIRED FILE MAPPINGS above using a bulleted list. 
   Your reply MUST end with the exact string "[UPLOAD_REQUIRED]". Do nothing else.
2. STEP 2 (Artifact Validation): If artifacts are present, look at the filenames in `Artifacts uploaded so far`. 
   DO NOT CALL ANY TOOLS YET. First, verify that the uploaded filenames match the REQUIRED FILE MAPPINGS for the user's issue.
   If the required files are MISSING or INCORRECT, you MUST tell the user what is missing and output the exact string "[UPLOAD_REQUIRED]" again. Do not call tools.
3. STEP 3 (Tool Usage): Only if the correct files are verified in STEP 2, use your diagnostic tools to evaluate the uploaded files. 
4. STEP 4 (Issue Found): If your tool results show an issue, you MUST ask the user for permission to investigate deeper. Your reply MUST contain the exact string "ACTION: REQUIRE_APPROVAL".
5. STEP 5 (Approval Granted): If the user approves, provide the final root cause and the exact code or fix needed.

Do NOT output "ACTION: REQUIRE_APPROVAL" before you have run your diagnostic tools.
"""

def agent_node(state: AgentState):
    messages = state.get("messages", [])
    artifacts = state.get("artifacts", [])
    
    # Prepend system message
    sys_msg = SystemMessage(content=system_prompt.format(artifacts=", ".join(artifacts) if artifacts else "None"))
    
    # Only allow tool usage if there are actually artifacts to analyze!
    # This prevents small local models from hallucinating tool inputs before files exist.
    if not artifacts:
        response = llm.invoke([sys_msg] + messages)
        if "[UPLOAD_REQUIRED]" not in response.content:
            response = AIMessage(content=response.content + "\n\nPlease upload the necessary files. [UPLOAD_REQUIRED]")
    else:
        response = llm_with_tools.invoke([sys_msg] + messages)
        # Defensive programming if the LLM stalls
        if not response.tool_calls and "[UPLOAD_REQUIRED]" not in response.content and "ACTION: REQUIRE_APPROVAL" not in response.content:
            response = AIMessage(content=response.content + "\n\nI need you to upload your files so I can check. [UPLOAD_REQUIRED]")
        
    requires_approval = state.get("requires_user_approval", False)
    
    # Check if we should pause for approval
    if response.content and "ACTION: REQUIRE_APPROVAL" in response.content:
        # We replace the text to be user friendly
        response = AIMessage(content=response.content.replace("ACTION: REQUIRE_APPROVAL", "") + "\n\nI detected an issue. Would you like me to continue deeper investigation?")
        requires_approval = True
        
    return {"messages": [response], "requires_user_approval": requires_approval}

tool_node = ToolNode(tools)

def should_continue(state: AgentState) -> Literal["tools", "human_approval", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Safely check for tool calls (HumanMessage doesn't have this attribute)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    # If LLM requested approval, route to human
    if state.get("requires_user_approval", False) and not state.get("investigation_approved", False):
        return "human_approval"
        
    return "__end__"

def human_approval_node(state: AgentState):
    # This node is a placeholder that will be interrupted before execution.
    # In LangGraph, we interrupt before this node.
    pass

# Build graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("human_approval", human_approval_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "human_approval": "human_approval",
    "__end__": END
})

workflow.add_edge("tools", "agent")
workflow.add_edge("human_approval", "agent")

# We interrupt before human_approval
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory, interrupt_before=["human_approval"])
