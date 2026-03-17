# AiDa: AI Diagnostic Agent for ML Pipelines

AiDa is an interactive, locally hosted AI agent designed to troubleshoot and diagnose complex Machine Learning pipeline errors. Whether you are dealing with "JSON score mismatches" or "Model load failures," AiDa acts as a smart debugging assistant to pinpoint root causes, execute diagnostic tests, and provide code-level fixes with a secure, human-in-the-loop workflow.

## 🚀 Key Features

- **Smart Artifact Extraction:** AiDa correctly identifies the exact files needed to debug your specific issue based on strict signature mappings.
- **Automated Validation:** Prevents misdiagnosis and LLM hallucination by verifying the presence and names of required files before executing any diagnostic tools.
- **Sandboxed File Handling:** Securely stores all user uploads in timestamp-labelled sandbox directories (e.g., `sandbox/YYYYMMDD_HHMMSS_<thread_id>`), allowing precise tracing of tool executions.
- **Human-in-the-Loop Safeguards:** Features an interactive approval system. If an anomaly is detected, AiDa halts execution and explicitly asks for your permission before diving into deep root-cause investigations.

## 🧠 Architecture & Tech Stack

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) handling the web server, file uploads, and API endpoints (`/chat`, `/upload`, `/approve`).
- **Frontend:** Vanilla HTML/JS served statically.
- **LLM Engine:** Local models powered by [Ollama](https://ollama.ai/) (defaulting to `llama3.2`).
- **Agent Orchestration:** [LangGraph](https://python.langchain.com/docs/langgraph/) for stateful, cyclical, and interruptible multi-agent workflows.
- **State Management:** LangGraph's `MemorySaver` preserves context and handles state variables (`messages`, `artifacts`, `requires_user_approval`, `investigation_approved`) across sessions.

## 🛠️ Built-In Diagnostic Tools

AiDa comes equipped with specialized Python tools to diagnose ML pipelines:
- `validate_schema`: Assesses the baseline shape of datasets against expected schemas to find anomalies quickly.
- `score_model`: Evaluates models on provided datasets to catch drops in target predictions.
- `test_model_load`: Attempts mock model structure mounting, finding missing "layers" or configurations inside `JSON` architecture maps.
- `compare_scoring_pipelines`: Compares inference scripts against reference standards to unearth preprocessing mismatches (e.g., missing normalization).

## ⚡ Workflow
1. **Describe the Issue:** User states the problem. AiDa identifies required files and prompts the UI for an upload.
2. **Upload Artifacts:** User uploads exactly the files requested. AiDa validates file mapping before proceeding.
3. **Execute Diagnostics:** AiDa runs applicable test tools on your data.
4. **Human Approval:** If a discrepancy is found, execution pauses. The user is asked to grant permission for deeper investigation.
5. **Final Diagnosis:** Execution resumes, and AiDa delivers the exact error root cause alongside a practical code fix.

## Architecture: 


## 💻 Setup & Installation

**Prerequisites:**
- [Ollama](https://ollama.ai/) installed and running on your local machine.
- [uv](https://github.com/astral-sh/uv) (Python packaging and execution).

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd ai_diagnostic_agent
   ```

2. **Environment Variables:**
   Copy the example `.env` file to set up your environment (e.g., configuring `OLLAMA_MODEL`):
   ```bash
   cp .env.example .env
   ```

3. **Start the Development Server:**
   Use the included Makefile to quickly spin up the backend:
   ```bash
   make dev
   ```
   This runs the FastAPI app via `uv run uvicorn src.server:app --reload --port 8000`.

4. **Access the App:**
   Open your browser and navigate to [http://localhost:8000](http://localhost:8000).

---
*Note: This agent relies on LangGraph's interrupted execution paradigm. Ensure you do not skip the approval steps when interacting with complex, system-modifying diagnostics.*
