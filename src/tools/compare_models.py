import json
import xml.etree.ElementTree as ET

def compare_json_pmml(json_model_path, pmml_path):
    """
    Compares structural differences between JSON and PMML models.
    """

    try:
        # ---- Load JSON Model ----
        with open(json_model_path, 'r') as f:
            json_model = json.load(f)

        json_trees = json_model.get("learner", {}) \
                               .get("gradient_booster", {}) \
                               .get("model", {}) \
                               .get("trees", [])

        json_tree_count = len(json_trees)

        # Extract JSON features
        json_features = set()
        for tree in json_trees:
            for node in tree.get("nodes", []):
                if "split" in node:
                    json_features.add(node["split"])

        # ---- Load PMML Model ----
        tree = ET.parse(pmml_path)
        root = tree.getroot()

        # Count trees in PMML
        pmml_trees = root.findall(".//TreeModel")
        pmml_tree_count = len(pmml_trees)

        # Extract PMML features
        pmml_features = set()
        for node in root.findall(".//Node"):
            if 'field' in node.attrib:
                pmml_features.add(node.attrib['field'])

        # ---- Comparison ----
        return {
            "json_tree_count": json_tree_count,
            "pmml_tree_count": pmml_tree_count,
            "tree_count_match": json_tree_count == pmml_tree_count,

            "json_features": list(json_features),
            "pmml_features": list(pmml_features),

            "feature_overlap": list(json_features.intersection(pmml_features)),
            "feature_mismatch": list(json_features.symmetric_difference(pmml_features))
        }

    except Exception as e:
        return {"error": str(e)}