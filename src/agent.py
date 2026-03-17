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
    demo_step = state.get("demo_step", 0)
    requires_approval = state.get("requires_user_approval", False)

    # Get last user message
    last_user_msg = ""
    for msg in reversed(messages):
        if getattr(msg, 'type', '') == 'human' or (isinstance(msg, tuple) and msg[0] == "user"):
            last_user_msg = msg[1] if isinstance(msg, tuple) else getattr(msg, 'content', '')
            break
            
    last_user_msg_lower = last_user_msg.lower() if isinstance(last_user_msg, str) else ""

    if demo_step == 0:
        if "upload" in last_user_msg_lower and artifacts:
            # Validate 4 files
            has_dataset = any("dataset" in f.lower() or "data" in f.lower() for f in artifacts)
            has_header = any("header" in f.lower() for f in artifacts)
            has_model = any("model" in f.lower() or "json" in f.lower() for f in artifacts)
            has_scores = sum(1 for f in artifacts if "score" in f.lower()) >= 2
            
            if len(artifacts) >= 4 and has_model: 
                demo_step = 1
                script_code = '```python\nimport pandas as pd\nimport numpy as np\n\ndef compare_scores(swift_path, open_source_path):\n    swift_df = pd.read_csv(swift_path)\n    open_df = pd.read_csv(open_source_path)\n    \n    # Calculate score deltas\n    diffs = swift_df["score"] - open_df["score"]\n    diffs.to_csv("score_diffs.csv", index=False)\n    return diffs\n\nif __name__ == "__main__":\n    compare_scores("swift_scores.csv", "open_source_scores.csv")\n```'
                response = AIMessage(content=f"Thanks for uploading the files, I have created a python script to see if the scores match. \n\n{script_code}\n\nDo you allow me to run it?\n\nACTION: REQUIRE_APPROVAL")
                return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}
            else:
                response = AIMessage(content="I see some files, but I need all of them to proceed. Please ensure you upload the dataset, header, model, and both score files (swift and open-source). [UPLOAD_REQUIRED]")
                return {"messages": [response], "demo_step": 0}
        else:
            response = AIMessage(content="Please upload the following files so I can compare:\n\n- Dataset file (e.g. data.csv)\n- Header file (e.g. header.csv)\n- XGBoost Model file (e.g. model.json)\n- Swift pipeline scores\n- Open-source pipeline scores (e.g. swift_scores.csv, open_source_scores.csv)\n\n[UPLOAD_REQUIRED]")
            return {"messages": [response], "demo_step": 0}
            
    elif demo_step == 1:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 2
            response = AIMessage(content="Confirmed: scores differ. Example deltas (swift - open): \nRow 0: 0.000000012\nRow 1: -0.000000008\nRow 2: 0.000000015\n\nHere is the score diff file: [Download score_diffs.csv]({sandbox_dir}/score_diffs.csv).\nDo you want me to investigate further?")
            return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
        
    elif demo_step == 2:
        if "yes" in last_user_msg_lower or "investigate" in last_user_msg_lower:
            demo_step = 3
            response = AIMessage(content="Next, I need per-tree leaf outputs for a small set (e.g., 10 records). May I generate tree-level scores for both pipelines?\n\nACTION: REQUIRE_APPROVAL")
            return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}

    elif demo_step == 3:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 4
            script_code = '```python\nimport xgboost as xgb\nimport pandas as pd\n\ndef trace_decision_paths(model_path, data_path):\n    # Load XGBoost model and track tree nodes\n    model = xgb.Booster()\n    model.load_model(model_path)\n    dmatrix = xgb.DMatrix(data_path)\n    \n    # Dump path for debugging precision\n    paths = model.predict(dmatrix, pred_leaf=True)\n    pd.DataFrame(paths).to_csv("paths.csv", index=False)\n    return paths\n```'
            response = AIMessage(content=f"Tree-level outputs generated. I'll now trace decision paths to find divergence points. \n[Download swift_tree_scores.csv]({{sandbox_dir}}/swift_tree_scores.csv) \n[Download open_source_tree_scores.csv]({{sandbox_dir}}/open_source_tree_scores.csv).\n\nI'll create a paths log comparing swift vs. open-source per tree using the following script, including the first differing node and feature value:\n\n{script_code}\n\nProceed?\n\nACTION: REQUIRE_APPROVAL")
            return {"messages": [response], "requires_user_approval": True, "demo_step": demo_step}

    elif demo_step == 4:
        if "approve" in last_user_msg_lower or "yes" in last_user_msg_lower:
            demo_step = 5
            response = AIMessage(content="Found divergences. The first differing split often occurs at feature X with thresholds very close to the record's value (precision-sensitive).\n\nConclusion: Path divergence seems due to precision differences (values near split thresholds) caused the score mismatch. Recommend aligning numeric precision/threshold handling between swift and open-source.\n[Download paths.csv]({sandbox_dir}/paths.csv)")
            return {"messages": [response], "requires_user_approval": False, "demo_step": demo_step}
            
    if demo_step >= 5:
        demo_step = 6

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
