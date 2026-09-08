"""
Explainability Module
---------------------
Model interpretation and feature importance analysis (e.g., SHAP, LIME, Feature Importances).
"""

from typing import Any
import pandas as pd


def get_feature_importances(model: Any, feature_names: list) -> pd.DataFrame:
    """Extract and rank feature importances if supported by the model."""
    if hasattr(model, "feature_importances_"):
        df_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values(by="importance", ascending=False)
        return df_importance
    return pd.DataFrame(columns=["feature", "importance"])


if __name__ == "__main__":
    print("Explainability module initialized.")
