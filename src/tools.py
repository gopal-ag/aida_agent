from langchain_core.tools import tool
from typing import Dict, Any

@tool
def validate_schema(dataset_path: str, schema_path: str) -> str:
    """Validates the dataset schema against expected schema. Use this first when investigating dataset issues."""
    return f"Schema Validation [Dataset: {dataset_path}, Schema: {schema_path}]: PASSED. Expected fields match dataset."

@tool
def compare_scoring_pipelines(script_path: str, reference_path: str) -> str:
    """Compares the user's inference script against a reference baseline. Use this to investigate 'JSON score mismatch'."""
    return f"ISSUE DETECTED in '{script_path}': Preprocessing mismatch. The inference script does not normalize the 'income' feature before making predictions."

@tool
def score_model(model_path: str, dataset_path: str) -> str:
    """Runs a quick evaluation of the model on the provided dataset."""
    return f"Model scoring completed for '{model_path}' on '{dataset_path}'. Output prediction mismatch detected: Expected mean score=0.85, Actual mean score=0.62. Potential cause: Preprocessing differences or schema mismatch."

@tool
def test_model_load(model_path: str, config_path: str) -> str:
    """Tests loading the JSON model architecture and weights. Use this to investigate issues where the model 'cannot be loaded' or throws a parsing error."""
    return f"ISSUE DETECTED inside {model_path}: Architecture key missing. The JSON model file is missing the 'layers' configuration array required by {config_path}."
