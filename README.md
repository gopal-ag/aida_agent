# AiDa: AI Diagnostic Agent for ML Pipelines

AiDa is an interactive, locally hosted AI agent designed to troubleshoot and diagnose complex Machine Learning pipeline errors. Whether you are dealing with "JSON score mismatches" or "Model load failures," AiDa acts as a smart debugging assistant to pinpoint root causes, execute diagnostic tests, and provide code-level fixes with a secure, human-in-the-loop workflow.
(before single output)
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
<img width="1414" height="2000" alt="arc" src="https://github.com/user-attachments/assets/c1de8e95-bebb-4757-b42e-f4c2b19e4f71" />

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








Promt: i want you to comepletly change the flow of agent and make it a demo to showcase solving of one specific issue the frontend will reamin same but i want you to follow this story line: 

context for the demo: User scored 10 records with internal pipeline “swift” and a public “open-source” scorer. Both used the same XGBoost model (350 features, 1000 rows  data;

Observation: Final scores differed slightly between swift and open-source (Cell 11 scores CSVs).

Diagnostic step 1 — Tree-level scoring: User generated per-tree leaf outputs for the same 10 records (Cell 11) to see where paths might diverge.

Diagnostic step 2 — Path tracing: User built a paths log with 430 entries (Cell 12) showing, per tree, the sequence of split decisions for swift vs. open-source. Each path lists at least 15 split conditions, the first differing node, and the feature value at that node.

Root cause: In the path log, the first differing node often had a value extremely close to the split threshold (9-decimal precision), indicating tiny precision differences caused the branch divergence and the score mismatch.

i have already created and saved all the files needed for this experiment under /data 

this is the kind of story line you can expect and i want the agent to only stick to this and nothing else ( its a very small model so make sure resposnses are always same and does not helluciante due to too many isturcions) 

User: “Scores are not matching between swift and open-source.”
Bot: “Please upload the dataset, header, model, and both score files (swift and open-source) so I can compare.”
User uploads files.
Bot: thanks for uploading the files, i have created the below python script to see if the scores match do you allow me to run it?
user: approved: 
Bot runs a quick diff (show a demo with loading ui) : “Confirmed: scores differ. Example deltas (swift – open): … (shows small sample). 
give the score_diff.csv for download to the user in ui 
Do you want me to investigate further?”
User: “Yes.”
Bot: “Next, I need per-tree leaf outputs for a small set (e.g., 10 records). May I generate tree-level scores for both pipelines?”
User: “Yes.”
Bot generates tree-level scores: “Tree-level outputs generated. I’ll now trace decision paths to find divergence points.”
give these artifacts in dowload in ui as well 
Bot: “I’ll create a paths log comparing swift vs. open-source per tree, including the first differing node and feature value. Proceed?”
User: “Yes.”
Bot produces paths log: “Found divergences. The first differing split often occurs at feature X with thresholds very close to the record’s value (precision-sensitive).”
Bot: “Conclusion: Path divergence seems due to precision differences (values near split thresholds) caused the score mismatch. Recommend aligning numeric precision/threshold handling between swift and open-source.” give artifact to download 
User: “Got it.”

so in frontend you only need to create the new donload ui, for sandox i want you to that when when user uploads you store as it already does, but for eg when files are genrated (demo) eg score_diff.json you copy that score_diff.csv from data folder to that perticular sandbox and also make it availbale for download via ui same for other files and, also create dummy but real looking py scirpts for score comparison and path genration make it follow the exact behaviour and only make it for this isssue and reduce dependnce on llm so the model doesnt hellusicante and this demo works excatly as it is 