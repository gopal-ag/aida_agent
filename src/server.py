from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os
from datetime import datetime
from langchain_core.messages import HumanMessage

from src.agent import graph

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class ApproveRequest(BaseModel):
    thread_id: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("static/index.html", "r") as f:
        return f.read()

@app.post("/chat")
async def chat(request: ChatRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Run the graph
    events = graph.stream(
        {"messages": [("user", request.message)]}, 
        config, 
        stream_mode="values"
    )
    
    final_state = None
    for event in events:
        final_state = event
        
    state = graph.get_state(config)
    messages = state.values.get("messages", [])
    
    if not messages:
        return {"response": "No response"}
        
    last_msg = messages[-1].content
    
    # Check if pausing for approval
    requires_approval = False
    if state.next and state.next[0] == "human_approval":
        requires_approval = True
        
    upload_required = "[upload_required]" in last_msg.lower()
    
    # Clean up markers for display using case-insensitive replace
    import re
    display_msg = re.sub(r'(?i)\[UPLOAD_REQUIRED\]', '', last_msg).strip()
    
    return {
        "response": display_msg,
        "requires_approval": requires_approval,
        "upload_required": upload_required
    }

@app.post("/upload")
async def upload_artifacts(thread_id: str, files: List[UploadFile] = File(...)):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Create timestamped sandbox directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sandbox_dir = os.path.join("sandbox", f"{timestamp}_{thread_id}")
    os.makedirs(sandbox_dir, exist_ok=True)
    
    saved_paths = []
    filenames = []
    
    for file in files:
        file_path = os.path.join(sandbox_dir, file.filename)
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        saved_paths.append(file_path)
        filenames.append(file.filename)
    
    # Update Agent state
    state = graph.get_state(config)
    current_artifacts = state.values.get("artifacts", []) if state.values else []
    
    # We append exact filenames so the LLM's prompt logic matches exactly against REQUIRED MAPPINGS
    current_artifacts.extend(filenames)
    
    upload_msg = "SYSTEM NOTIFICATION: The user has uploaded the following files to the sandbox:\n" + "\n".join([f"- {name} (Path: {path})" for name, path in zip(filenames, saved_paths)]) + "\nProceed to STEP 2 (Artifact Validation) immediately."
    
    # We only pass the NEW message, let LangGraph's add_messages reducer handle the rest
    graph.update_state(config, {
        "artifacts": current_artifacts, 
        "messages": [HumanMessage(content=upload_msg)]
    })
    
    return {"message": f"Successfully uploaded {len(files)} files to {sandbox_dir}.", "artifacts": current_artifacts}

@app.post("/approve")
async def approve_investigation(request: ApproveRequest):
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Update state to approve
    graph.update_state(config, {"investigation_approved": True, "requires_user_approval": False})
    
    # Resume execution
    events = graph.stream(None, config, stream_mode="values")
    
    final_state = None
    for event in events:
        final_state = event
        
    state = graph.get_state(config)
    last_msg = state.values.get("messages", [])[-1].content
    
    return {"response": last_msg}
