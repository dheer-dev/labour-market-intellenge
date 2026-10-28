from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset_occupation_aware.csv"
FOLD_OUTPUT = PROCESSED_DIR / "model_occupation_aware_baseline_fold_metrics.csv"
SUMMARY_OUTPUT = PROCESSED_DIR / "model_occupation_aware_baseline_summary.csv"


def make_text_model():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def metrics(y_true, predictions, probabilities):
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "occupation_uri",
        "occupation_label",
        "skill_label",
        "skill_type",
        "relation_type",
        "market_observed",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[["occupation_uri", "market_observed"]].isna().any().any():
        raise ValueError("Occupation IDs or target labels are missing.")

    if df.duplicated(["occupation_uri", "skill_uri"]).any():
        raise ValueError("Duplicate occupation-skill pairs detected.")

    df["market_observed"] = df["market_observed"].astype(int)
    if set(df["market_observed"].unique()) != {0, 1}:
        raise ValueError("The target must contain both classes.")

    text_columns = [
        "occupation_label",
        "skill_label",
        "skill_type",
        "relation_type",
    ]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    X = df[text_columns].agg(" ".join, axis=1).str.strip()
    y = df["market_observed"]
    groups = df["occupation_uri"]

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(f"Occupation leakage in fold {fold}.")

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(f"Fold {fold} is missing a class.")

        # Baseline: always predict the majority class in training.
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(X.iloc[train_idx].to_frame(), y_train)

        baseline_pred = baseline.predict(X.iloc[test_idx].to_frame())
        baseline_prob = baseline.predict_proba(
            X.iloc[test_idx].to_frame()
        )[:, 1]

        # Text model: same TF-IDF + Logistic Regression settings.
        model = make_text_model()
        model.fit(X.iloc[train_idx], y_train)

        model_pred = model.predict(X.iloc[test_idx])
        model_prob = model.predict_proba(X.iloc[test_idx])[:, 1]

        for name, pred, prob in [
            ("Majority-class baseline", baseline_pred, baseline_prob),
            ("TF-IDF + Logistic Regression", model_pred, model_prob),
        ]:
            rows.append({
                "fold": fold,
                "model": name,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_occupations": len(train_groups),
                "test_occupations": len(test_groups),
                "occupation_overlap": len(train_groups.intersection(test_groups)),
                **metrics(y_test, pred, prob),
            })

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(FOLD_OUTPUT, index=False)

    metric_names = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
    ]
    summary_rows = []

    for model_name, subset in fold_df.groupby("model", sort=False):
        for metric in metric_names:
            summary_rows.append({
                "model": model_name,
                "metric": metric,
                "mean": subset[metric].mean(),
                "std": subset[metric].std(ddof=1),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUTPUT, index=False)

    print("=" * 72)
    print("BASELINE VS TEXT MODEL — GROUPED CROSS-VALIDATION")
    print("=" * 72)
    print("Dataset rows:", len(df))
    print("Unique occupations:", groups.nunique())
    print("Folds:", N_SPLITS)
    print("Occupation overlap in every fold: 0")
    print("\nFold-level results:")
    print(fold_df.to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))

    print("\nMean ± standard deviation across folds:")
    for model_name, subset in fold_df.groupby("model", sort=False):
        print(f"\n{model_name}")
        for metric in metric_names:
            print(
                f"  {metric:10s}: {subset[metric].mean():.4f} "
                f"± {subset[metric].std(ddof=1):.4f}"
            )

    print("\nSaved:")
    print(FOLD_OUTPUT)
    print(SUMMARY_OUTPUT)
    print("\nOriginal datasets, models, and results were not modified.")


if __name__ == "__main__":
    main()
