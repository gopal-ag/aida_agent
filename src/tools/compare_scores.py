import pandas as pd
import numpy as np
import os

def run_score_comparison(swift_path, os_path, output_path="/sandbox/score_diff.csv"):
    """
    Compares scores between Swift (Internal) and Open-Source XGBoost pipelines.
    Identifies precision-based deltas.
    """
    try:
        # Load datasets
        df_swift = pd.read_csv(swift_path)
        df_os = pd.read_csv(os_path)

        # Ensure alignment on ID or Index
        if 'id' in df_swift.columns:
            df_swift = df_swift.set_index('id')
            df_os = df_os.set_index('id')

        # Create comparison matrix
        comparison = pd.DataFrame(index=df_swift.index)
        comparison['swift_score'] = df_swift['score']
        comparison['os_score'] = df_os['score']
        
        # Calculate Absolute and Relative Difference
        comparison['abs_diff'] = (comparison['swift_score'] - comparison['os_score']).abs()
        comparison['rel_diff'] = comparison['abs_diff'] / (comparison['os_score'] + 1e-9)
        
        # Filter for non-matching records (Tolerance at 1e-7)
        mismatches = comparison[comparison['abs_diff'] > 1e-8].copy()
        
        # Save to sandbox for user download
        mismatches.to_csv(output_path)
        
        return {
            "total_records": len(comparison),
            "mismatch_count": len(mismatches),
            "max_delta": mismatches['abs_diff'].max() if not mismatches.empty else 0,
            "artifact": output_path
        }
    except Exception as e:
        return {"error": str(e)}