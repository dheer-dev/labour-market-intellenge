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
OUTER_SPLITS = 5
INNER_SPLITS = 4
THRESHOLDS = np.round(np.arange(0.20, 0.651, 0.05), 2)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = DATA_DIR / "model_dataset_occupation_aware.csv"
FOLD_FILE = DATA_DIR / "nested_threshold_fold_results.csv"
PREDICTION_FILE = DATA_DIR / "nested_threshold_oof_predictions.csv"
SUMMARY_FILE = DATA_DIR / "nested_threshold_summary.csv"


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


def calculate_metrics(y_true, probability, threshold):
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    return {
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


def choose_threshold(y_true, probability):
    """Choose threshold using training-only inner OOF predictions."""
    results = []

    for threshold in THRESHOLDS:
        prediction = (probability >= threshold).astype(int)
        results.append({
            "threshold": float(threshold),
            "f1": f1_score(y_true, prediction, zero_division=0),
            "precision": precision_score(
                y_true, prediction, zero_division=0
            ),
            "recall": recall_score(
                y_true, prediction, zero_division=0
            ),
        })

    # Maximise F1. In a tie, prefer higher precision, then higher threshold.
    best = max(
        results,
        key=lambda row: (row["f1"], row["precision"], row["threshold"]),
    )
    return best, results


def main():
    df = pd.read_csv(INPUT_FILE)

    required = [
        "occupation_uri",
        "occupation_label",
        "skill_uri",
        "skill_label",
        "skill_type",
        "relation_type",
        "market_observed",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[["occupation_uri", "skill_uri", "market_observed"]].isna().any().any():
        raise ValueError("Missing occupation IDs, skill IDs, or target labels.")

    if df.duplicated(["occupation_uri", "skill_uri"]).any():
        raise ValueError("Duplicate occupation-skill pairs found.")

    df["market_observed"] = df["market_observed"].astype(int)
    if set(df["market_observed"].unique()) != {0, 1}:
        raise ValueError("Expected both target classes: 0 and 1.")

    text_columns = [
        "occupation_label", "skill_label", "skill_type", "relation_type"
    ]
    for column in text_columns:
        df[column] = df[column].fillna("").astype(str)

    X = df[text_columns].agg(" ".join, axis=1).str.strip()
    y = df["market_observed"].reset_index(drop=True)
    groups = df["occupation_uri"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    df = df.reset_index(drop=True)

    group_class_counts = df.groupby("occupation_uri")[
        "market_observed"
    ].nunique()
    if not (group_class_counts == 2).all():
        raise ValueError(
            "Every occupation must contain both classes for grouped CV."
        )

    outer_cv = StratifiedGroupKFold(
        n_splits=OUTER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    outer_splits = list(outer_cv.split(X, y, groups=groups))

    print("Nested occupation-aware threshold evaluation")
    print("Input:", INPUT_FILE)
    print("Rows:", len(df))
    print("Occupations:", groups.nunique())
    print("Class counts:", y.value_counts().sort_index().to_dict())
    print("Outer folds:", OUTER_SPLITS, "| Inner folds:", INNER_SPLITS)
    print("Candidate thresholds:", THRESHOLDS.tolist())

    fold_results = []
    prediction_frames = []

    for model_name, class_weight in [
        ("baseline", None),
        ("balanced", "balanced"),
    ]:
        print(f"\n{'=' * 70}\nMODEL: {model_name}\n{'=' * 70}")

        for outer_fold, (train_idx, test_idx) in enumerate(
            outer_splits, start=1
        ):
            X_train = X.iloc[train_idx].reset_index(drop=True)
            y_train = y.iloc[train_idx].reset_index(drop=True)
            groups_train = groups.iloc[train_idx].reset_index(drop=True)

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            test_groups = set(groups.iloc[test_idx])
            train_groups = set(groups_train)

            if train_groups.intersection(test_groups):
                raise RuntimeError(
                    f"Occupation leakage in outer fold {outer_fold}"
                )

            # Inner grouped OOF probabilities are used ONLY to select threshold.
            inner_cv = StratifiedGroupKFold(
                n_splits=INNER_SPLITS,
                shuffle=True,
                random_state=RANDOM_STATE + outer_fold,
            )
            inner_probability = np.full(len(X_train), np.nan)

            for inner_train_idx, inner_valid_idx in inner_cv.split(
                X_train, y_train, groups=groups_train
            ):
                inner_model = build_model(class_weight)
                inner_model.fit(
                    X_train.iloc[inner_train_idx],
                    y_train.iloc[inner_train_idx],
                )
                inner_probability[inner_valid_idx] = (
                    inner_model.predict_proba(
                        X_train.iloc[inner_valid_idx]
                    )[:, 1]
                )

            if np.isnan(inner_probability).any():
                raise RuntimeError(
                    f"Missing inner OOF predictions in outer fold {outer_fold}"
                )

            best, _ = choose_threshold(y_train, inner_probability)
            selected_threshold = best["threshold"]

            # Refit on the entire outer training partition.
            final_fold_model = build_model(class_weight)
            final_fold_model.fit(X_train, y_train)
            test_probability = final_fold_model.predict_proba(X_test)[:, 1]

            selected_metrics = calculate_metrics(
                y_test, test_probability, selected_threshold
            )
            fixed_metrics = calculate_metrics(
                y_test, test_probability, 0.50
            )

            fold_results.append({
                "model": model_name,
                "outer_fold": outer_fold,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_occupations": len(train_groups),
                "test_occupations": len(test_groups),
                "selected_threshold": selected_threshold,
                **{
                    f"selected_{key}": value
                    for key, value in selected_metrics.items()
                },
                **{
                    f"fixed_050_{key}": value
                    for key, value in fixed_metrics.items()
                },
            })

            prediction_frames.append(pd.DataFrame({
                "occupation_uri": df.iloc[test_idx]["occupation_uri"].values,
                "occupation_label": df.iloc[test_idx]["occupation_label"].values,
                "skill_uri": df.iloc[test_idx]["skill_uri"].values,
                "skill_label": df.iloc[test_idx]["skill_label"].values,
                "market_observed": y_test.values,
                "model": model_name,
                "outer_fold": outer_fold,
                "selected_threshold": selected_threshold,
                "predicted_probability": test_probability,
            }))

            print(
                f"Fold {outer_fold}: threshold={selected_threshold:.2f} | "
                f"F1={selected_metrics['f1']:.3f} | "
                f"Precision={selected_metrics['precision']:.3f} | "
                f"Recall={selected_metrics['recall']:.3f} | "
                f"PR-AUC={selected_metrics['pr_auc']:.3f}"
            )

    folds = pd.DataFrame(fold_results)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    # Calculate pooled outer-fold metrics using each fold's training-only threshold.
    summary_rows = []
    for model_name, group in predictions.groupby("model"):
        y_true = group["market_observed"].astype(int).to_numpy()
        probability = group["predicted_probability"].to_numpy()
        selected = (
            probability >= group["selected_threshold"].to_numpy()
        ).astype(int)

        fixed = (probability >= 0.50).astype(int)

        for evaluation, pred in [
            ("nested_selected_threshold", selected),
            ("fixed_threshold_050", fixed),
        ]:
            tn, fp, fn, tp = confusion_matrix(
                y_true, pred, labels=[0, 1]
            ).ravel()
            values = {
                "accuracy": accuracy_score(y_true, pred),
                "precision": precision_score(
                    y_true, pred, zero_division=0
                ),
                "recall": recall_score(
                    y_true, pred, zero_division=0
                ),
                "f1": f1_score(y_true, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_true, probability),
                "pr_auc": average_precision_score(y_true, probability),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
            }
            for metric_name, value in values.items():
                summary_rows.append({
                    "model": model_name,
                    "evaluation": evaluation,
                    "metric": metric_name,
                    "value": value,
                })

        selected_fold = folds[folds["model"] == model_name]
        summary_rows.append({
            "model": model_name,
            "evaluation": "threshold_selection",
            "metric": "mean_selected_threshold",
            "value": selected_fold["selected_threshold"].mean(),
        })
        summary_rows.append({
            "model": model_name,
            "evaluation": "threshold_selection",
            "metric": "std_selected_threshold",
            "value": selected_fold["selected_threshold"].std(ddof=1),
        })

    summary = pd.DataFrame(summary_rows)
    folds.to_csv(FOLD_FILE, index=False)
    predictions.to_csv(PREDICTION_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\nPOOLED OUTER-FOLD RESULTS")
    print(
        summary[
            summary["evaluation"].isin([
                "nested_selected_threshold", "fixed_threshold_050"
            ])
        ].pivot(index=["model", "evaluation"], columns="metric", values="value")
        .to_string(float_format=lambda value: f"{value:.4f}")
    )

    print("\nSelected thresholds by fold")
    print(
        folds[["model", "outer_fold", "selected_threshold"]]
        .to_string(index=False)
    )

    print("\nSaved new files:")
    print(FOLD_FILE)
    print(PREDICTION_FILE)
    print(SUMMARY_FILE)
    print("Existing model files and evaluation outputs were not modified.")


if __name__ == "__main__":
    main()
