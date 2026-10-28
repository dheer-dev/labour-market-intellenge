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


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset_occupation_aware.csv"

FOLD_RESULTS_FILE = (
    PROCESSED_DIR / "model_occupation_aware_grouped_cv_fold_metrics.csv"
)
OOF_RESULTS_FILE = (
    PROCESSED_DIR / "model_occupation_aware_grouped_cv_oof_predictions.csv"
)
SUMMARY_FILE = (
    PROCESSED_DIR / "model_occupation_aware_grouped_cv_summary.csv"
)


def build_model():
    """Create the same TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
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
        ]
    )


def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate classification and ranking metrics."""
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc_score(y_true, y_probability),
        "pr_auc": average_precision_score(
            y_true, y_probability
        ),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
    }


def main():
    print("=" * 72)
    print("OCCUPATION-AWARE GROUPED CROSS-VALIDATION")
    print("=" * 72)

    # --------------------------------------------------
    # 1. Load and validate data
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "occupation_uri",
        "occupation_label",
        "skill_uri",
        "skill_label",
        "skill_type",
        "relation_type",
        "market_observed",
    ]

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[["occupation_uri", "skill_uri", "market_observed"]].isna().any().any():
        raise ValueError(
            "Missing values found in occupation identifiers, "
            "skill identifiers, or target labels."
        )

    if df.duplicated(["occupation_uri", "skill_uri"]).any():
        raise ValueError(
            "Duplicate occupation-skill pairs found."
        )

    df["market_observed"] = df["market_observed"].astype(int)

    if set(df["market_observed"].unique()) != {0, 1}:
        raise ValueError("Target must contain both classes: 0 and 1.")

    # Missing descriptive metadata is represented by empty text,
    # not the literal word "nan".
    text_columns = [
        "occupation_label",
        "skill_label",
        "skill_type",
        "relation_type",
    ]

    for col in text_columns:
        df[col] = df[col].fillna("").astype(str)

    df["model_text"] = df[text_columns].agg(" ".join, axis=1).str.strip()

    X = df["model_text"]
    y = df["market_observed"]
    groups = df["occupation_uri"]

    print("Dataset:", INPUT_FILE.name)
    print("Rows:", len(df))
    print("Unique occupations:", groups.nunique())
    print("Observed pairs:", int(y.sum()))
    print("Non-observed pairs:", int((y == 0).sum()))
    print("Missing values in text fields after handling:",
          int(df[text_columns].isna().sum().sum()))

    # Every included occupation should contain both classes.
    group_class_counts = df.groupby("occupation_uri")[
        "market_observed"
    ].nunique()

    if not (group_class_counts == 2).all():
        raise ValueError(
            "At least one occupation does not contain both classes."
        )

    # --------------------------------------------------
    # 2. Prepare grouped, stratified folds
    # --------------------------------------------------

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Store out-of-fold predictions. Each row is predicted only
    # by a model that was not trained on that occupation.
    oof_probability = np.full(len(df), np.nan)
    oof_prediction = np.full(len(df), -1, dtype=int)
    oof_fold = np.full(len(df), -1, dtype=int)

    fold_results = []

    # --------------------------------------------------
    # 3. Train and evaluate each fold
    # --------------------------------------------------

    for fold_number, (train_idx, test_idx) in enumerate(
        splitter.split(X, y, groups=groups),
        start=1,
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups.intersection(test_groups)

        if overlap:
            raise RuntimeError(
                f"Fold {fold_number} has occupation leakage: "
                f"{len(overlap)} overlapping occupations."
            )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                f"Fold {fold_number} does not contain both classes "
                "in training and testing."
            )

        model = build_model()
        model.fit(X.iloc[train_idx], y_train)

        probabilities = model.predict_proba(X.iloc[test_idx])[:, 1]
        predictions = (probabilities >= 0.50).astype(int)

        metrics = calculate_metrics(
            y_test,
            predictions,
            probabilities,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_occupations": len(train_groups),
                "test_occupations": len(test_groups),
                "occupation_overlap": len(overlap),
                "train_observed_pairs": int(y_train.sum()),
                "test_observed_pairs": int(y_test.sum()),
                "test_non_observed_pairs": int((y_test == 0).sum()),
                **metrics,
            }
        )

        oof_probability[test_idx] = probabilities
        oof_prediction[test_idx] = predictions
        oof_fold[test_idx] = fold_number

        print(
            f"\nFold {fold_number}/{N_SPLITS}: "
            f"train rows={len(train_idx)}, "
            f"test rows={len(test_idx)}, "
            f"train occupations={len(train_groups)}, "
            f"test occupations={len(test_groups)}, "
            f"overlap={len(overlap)}"
        )
        print(
            f"  Accuracy={metrics['accuracy']:.4f} | "
            f"Precision={metrics['precision']:.4f} | "
            f"Recall={metrics['recall']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f}"
        )

    # --------------------------------------------------
    # 4. Validate out-of-fold coverage
    # --------------------------------------------------

    if np.isnan(oof_probability).any():
        raise RuntimeError("Some rows did not receive OOF predictions.")

    if (oof_prediction == -1).any() or (oof_fold == -1).any():
        raise RuntimeError("Some rows are missing fold predictions.")

    fold_df = pd.DataFrame(fold_results)

    oof_df = df[
        [
            "occupation_uri",
            "occupation_label",
            "skill_uri",
            "skill_label",
            "market_observed",
        ]
    ].copy()

    oof_df["fold"] = oof_fold
    oof_df["predicted_probability"] = oof_probability
    oof_df["predicted_class_at_0_50"] = oof_prediction

    # --------------------------------------------------
    # 5. Fold averages and pooled OOF metrics
    # --------------------------------------------------

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary_rows = []

    for metric in metric_columns:
        summary_rows.append(
            {
                "result_type": "fold_mean",
                "metric": metric,
                "value": fold_df[metric].mean(),
                "fold_std": fold_df[metric].std(ddof=1),
            }
        )

    pooled_metrics = calculate_metrics(
        y,
        oof_prediction,
        oof_probability,
    )

    for metric in metric_columns:
        summary_rows.append(
            {
                "result_type": "pooled_oof",
                "metric": metric,
                "value": pooled_metrics[metric],
                "fold_std": np.nan,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # --------------------------------------------------
    # 6. Save new outputs only
    # --------------------------------------------------

    fold_df.to_csv(FOLD_RESULTS_FILE, index=False)
    oof_df.to_csv(OOF_RESULTS_FILE, index=False)
    summary_df.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 72)
    print("FOLD RESULTS")
    print("=" * 72)
    print(
        fold_df[
            [
                "fold",
                "train_occupations",
                "test_occupations",
                "occupation_overlap",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )

    print("\n" + "=" * 72)
    print("SUMMARY: MEAN AND STANDARD DEVIATION ACROSS FOLDS")
    print("=" * 72)

    for metric in metric_columns:
        mean = fold_df[metric].mean()
        std = fold_df[metric].std(ddof=1)
        print(f"{metric:12s}: {mean:.4f} ± {std:.4f}")

    print("\nPooled out-of-fold metrics:")
    for metric in metric_columns:
        print(f"{metric:12s}: {pooled_metrics[metric]:.4f}")

    print("\nSaved new files:")
    print(FOLD_RESULTS_FILE)
    print(OOF_RESULTS_FILE)
    print(SUMMARY_FILE)

    print("\nNo original datasets models, or results were modified.")


if __name__ == "__main__":
    main()
