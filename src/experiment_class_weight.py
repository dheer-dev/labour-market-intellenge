from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.50

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset_occupation_aware.csv"
SUMMARY_FILE = PROCESSED_DIR / "class_weight_experiment_summary.csv"
OOF_FILE = PROCESSED_DIR / "class_weight_experiment_oof.csv"


def build_model(class_weight):
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
                class_weight=class_weight,
            ),
        ),
    ])


def metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(
            y_true, prediction, zero_division=0
        ),
        "recall": recall_score(
            y_true, prediction, zero_division=0
        ),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
        "pr_auc": average_precision_score(y_true, probability),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
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
        raise ValueError(f"Missing columns: {missing}")

    if df[["occupation_uri", "market_observed"]].isna().any().any():
        raise ValueError("Missing occupation IDs or target labels.")

    if df.duplicated(["occupation_uri", "skill_uri"]).any():
        raise ValueError("Duplicate occupation-skill pairs found.")

    df["market_observed"] = df["market_observed"].astype(int)

    if set(df["market_observed"].unique()) != {0, 1}:
        raise ValueError("Expected target labels 0 and 1.")

    text_columns = [
        "occupation_label", "skill_label", "skill_type", "relation_type"
    ]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    X = df[text_columns].agg(" ".join, axis=1).str.strip()
    y = df["market_observed"]
    groups = df["occupation_uri"]

    group_class_counts = df.groupby("occupation_uri")[
        "market_observed"
    ].nunique()
    if not (group_class_counts == 2).all():
        raise ValueError(
            "Every occupation must contain both classes for this experiment."
        )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    splits = list(splitter.split(X, y, groups=groups))

    print("Input:", INPUT_FILE)
    print("Rows:", len(df))
    print("Occupations:", groups.nunique())
    print("Target counts:", y.value_counts().sort_index().to_dict())
    print("Threshold:", THRESHOLD)
    print("Models: baseline vs class_weight='balanced'")

    summary_rows = []
    oof_frames = []

    for model_name, class_weight in [
        ("baseline", None),
        ("balanced", "balanced"),
    ]:
        print(f"\n{'=' * 65}\nMODEL: {model_name}\n{'=' * 65}")

        oof_probability = np.full(len(df), np.nan)
        fold_rows = []

        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            train_groups = set(groups.iloc[train_idx])
            test_groups = set(groups.iloc[test_idx])
            overlap = train_groups.intersection(test_groups)
            if overlap:
                raise RuntimeError(f"Occupation leakage in fold {fold}")

            model = build_model(class_weight)
            model.fit(X.iloc[train_idx], y.iloc[train_idx])

            probability = model.predict_proba(X.iloc[test_idx])[:, 1]
            oof_probability[test_idx] = probability

            fold_metrics = metrics(
                y.iloc[test_idx], probability, THRESHOLD
            )
            fold_rows.append(fold_metrics)

            print(
                f"Fold {fold}: "
                f"F1={fold_metrics['f1']:.3f}, "
                f"Precision={fold_metrics['precision']:.3f}, "
                f"Recall={fold_metrics['recall']:.3f}, "
                f"PR-AUC={fold_metrics['pr_auc']:.3f}"
            )

        if np.isnan(oof_probability).any():
            raise RuntimeError("Missing out-of-fold probabilities.")

        fold_df = pd.DataFrame(fold_rows)
        pooled = metrics(y, oof_probability, THRESHOLD)

        for metric_name in [
            "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"
        ]:
            summary_rows.append({
                "model": model_name,
                "result_type": "fold_mean",
                "metric": metric_name,
                "value": fold_df[metric_name].mean(),
                "fold_std": fold_df[metric_name].std(ddof=1),
            })
            summary_rows.append({
                "model": model_name,
                "result_type": "pooled_oof",
                "metric": metric_name,
                "value": pooled[metric_name],
                "fold_std": np.nan,
            })

        for metric_name in [
            "threshold", "true_positive", "false_positive",
            "true_negative", "false_negative"
        ]:
            summary_rows.append({
                "model": model_name,
                "result_type": "pooled_oof",
                "metric": metric_name,
                "value": pooled[metric_name],
                "fold_std": np.nan,
            })

        oof_frames.append(pd.DataFrame({
            "occupation_uri": df["occupation_uri"],
            "occupation_label": df["occupation_label"],
            "skill_uri": df["skill_uri"],
            "skill_label": df["skill_label"],
            "market_observed": y,
            "model": model_name,
            "fold": [
                next(
                    fold_number
                    for fold_number, (_, test_idx) in enumerate(
                        splits, start=1
                    )
                    if row_index in set(test_idx)
                )
                for row_index in range(len(df))
            ],
            "predicted_probability": oof_probability,
            "predicted_class_at_0_50": (
                oof_probability >= THRESHOLD
            ).astype(int),
        }))

        print("\nPooled OOF metrics:")
        for key, value in pooled.items():
            print(f"{key:18s}: {value}")

    summary_df = pd.DataFrame(summary_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True)

    summary_df.to_csv(SUMMARY_FILE, index=False)
    oof_df.to_csv(OOF_FILE, index=False)

    print("\nSaved new experiment files:")
    print(SUMMARY_FILE)
    print(OOF_FILE)
    print("\nOriginal model files and grouped-CV results were not modified.")


if __name__ == "__main__":
    main()
