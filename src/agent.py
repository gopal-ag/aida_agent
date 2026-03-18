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

system_prompt = """You are AiDa, an expert AI diagnostic agent specialized in debugging and solving issues related to XGBoost models and ML pipelines.

Your goal is to quickly understand the user’s issue, ask for any missing artifacts if needed, and provide clear, practical solutions.

Guidelines:

If the issue lacks enough information, ask for only the relevant files or details (e.g., model file, config, sample data, logs).

Be adaptive, not rigid — don’t follow strict steps if unnecessary.

Prioritize fast diagnosis + actionable fixes over process.

When files are available, analyze them and explain findings clearly.

If unsure, ask smart follow-up questions instead of guessing.

Keep responses concise, helpful, and solution-oriented.

Style:

Think like a senior ML engineer debugging production issues.

Avoid unnecessary verbosity or strict workflows.

Focus on getting to the root cause quickly.
"""

def agent_node(state: AgentState):
    messages = state.get("messages", [])
    artifacts = state.get("artifacts", [])
    demo_step = state.get("demo_step", -1)
    requires_approval = state.get("requires_user_approval", False)

    # Get last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if getattr(msg, 'type', '') == 'human' or (isinstance(msg, tuple) and msg[0] == "user"):
            last_user_msg = msg[1] if isinstance(msg, tuple) else getattr(msg, 'content', '')
            break
            
    last_user_msg_lower = last_user_msg.lower() if isinstance(last_user_msg, str) else ""

    if demo_step == -1:
        if any(word in last_user_msg_lower for word in ["score", "scoring", "mismatch", "missmatch", "missmatach", "miss"]):
            demo_step = 0

    if demo_step == 0:
        if "upload" in last_user_msg_lower and artifacts:
            # Validate 4 files
            has_dataset = any("dataset" in f.lower() or "data" in f.lower() for f in artifacts)
            has_header = any("header" in f.lower() for f in artifacts)
            has_model = any("model" in f.lower() or "json" in f.lower() for f in artifacts)
            has_scores = sum(1 for f in artifacts if "score" in f.lower()) >= 2
            
            if len(artifacts) >= 4 and has_model: 
                demo_step = 1
                response = AIMessage(content="Thanks for uploading the files. Are you sure that the scores mismatch, or would you like me to check it for you?")
                return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
            else:
                response = AIMessage(content="I see some files, but I need all of them to proceed. Please ensure you upload the dataset, header, model, and both score files (swift and open-source). [UPLOAD_REQUIRED]")
                return {"messages": [response], "demo_step": 0}
        else:
            response = AIMessage(content="Please upload the following files so I can compare:\n\n- Dataset file (e.g. data.csv)\n- Header file (e.g. header.csv)\n- XGBoost Model file (e.g. model.json)\n- Swift pipeline scores\n- Open-source pipeline scores (e.g. swift_scores.csv, open_source_scores.csv)\n\n[UPLOAD_REQUIRED]")
            return {"messages": [response], "demo_step": 0}

    elif demo_step == 1:
        if "sure" in last_user_msg_lower or ("yes" in last_user_msg_lower and "check" not in last_user_msg_lower):
            demo_step = 3
            response = AIMessage(content="Confirmed: **scores differ**.\n\nDo you want me to investigate further?")
            return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
        else:
            demo_step = 2
            script_code = '```python\nimport pandas as pd\nimport numpy as np\n\ndef compare_scores(swift_path, open_source_path):\n    swift_df = pd.read_csv(swift_path)\n    open_df = pd.read_csv(open_source_path)\n    \n    # Calculate score deltas\n    diffs = swift_df["score"] - open_df["score"]\n    diffs.to_csv("score_diffs.csv", index=False)\n    return diffs\n\nif __name__ == "__main__":\n    compare_scores("swift_scores.csv", "open_source_scores.csv")\n```'
            response = AIMessage(content=f"I have created a python script to see if the scores match. \n\n{script_code}\n\nDo you allow me to run it?\n\nACTION: REQUIRE_APPROVAL")
            return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}
            
    elif demo_step == 2:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 3
            response = AIMessage(content="Confirmed: **scores differ**.\n\nSample differences (open - swift):\n\nacc_no,swift_score,open_score,diff_open_minus_swift\n**37.454011885**,717.882,713.379,**-4.502999999999929**\n**50.313625858**,471.684,476.006,**4.321999999999946**\n**53.258943255**,436.111,435.039,**-1.0720000000000027**\n\nHere is the full score diff file: [Download score_diffs.csv]({sandbox_dir}/score_diffs.csv).\n\nDo you want me to investigate further?")
            return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
        
    elif demo_step == 3:
        if "yes" in last_user_msg_lower or "investigate" in last_user_msg_lower:
            demo_step = 4
            response = AIMessage(content="Next, I need **per-tree leaf outputs** for a small set (e.g., 10 records). May I generate tree-level scores for both pipelines?\n\nACTION: REQUIRE_APPROVAL")
            return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}

    elif demo_step == 4:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 5
            response = AIMessage(content=f"Tree-level outputs generated. I'll now trace decision paths to find divergence points. \n[Download swift_tree_scores.csv]({{sandbox_dir}}/swift_tree_scores.csv) \n[Download open_source_tree_scores.csv]({{sandbox_dir}}/open_source_tree_scores.csv).\n\nI'll create a paths log comparing swift vs. open-source per tree using the following script, including the first differing node and feature value:\n\nProceed?\n\nACTION: REQUIRE_APPROVAL")
            return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}

    elif demo_step == 5:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 6
            response = AIMessage(content="Found divergences. The first differing split often occurs at feature **product_code < (97.32688)** with thresholds very close to the record's value **97.326875726** (precision-sensitive).\n\n**Conclusion:** Path divergence seems due to **precision differences** (values near split thresholds) caused the score mismatch. Recommend aligning numeric precision/threshold handling between swift and open-source.\n[Download All Artifacts]({sandbox_dir}/paths.csv)")
            return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
            
    if demo_step >= 6:
        demo_step = 7

    # Hybrid Fallback for regular queries / out-of-flow questions
    sys_msg = SystemMessage(content=system_prompt.format(artifacts=", ".join(artifacts) if artifacts else "None"))
    response = llm.invoke([sys_msg] + messages)
    
    if "ACTION: REQUIRE_APPROVAL" in response.content:
        response = AIMessage(content=response.content.replace("ACTION: REQUIRE_APPROVAL", "") + "\n\nI detected an issue. Would you like me to continue deeper investigation?")
        requires_approval = True
        
    return {"messages": [response], "requires_user_approval": requires_approval, "demo_step": demo_step}

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
