from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    artifacts: List[str]
    current_issue: Optional[str]
    requires_user_approval: bool
    investigation_approved: bool
    demo_step: int
