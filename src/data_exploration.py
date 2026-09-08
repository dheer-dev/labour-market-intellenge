"""
Data Exploration Module
-----------------------
Functions and scripts for exploratory data analysis (EDA) of labour market datasets.
"""

import pandas as pd
import numpy as np


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from path."""
    return pd.read_csv(file_path)


def basic_overview(df: pd.DataFrame) -> dict:
    """Return basic statistical and structural overview of the dataset."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


if __name__ == "__main__":
    print("Data exploration module initialized.")
