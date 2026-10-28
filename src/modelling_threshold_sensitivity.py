import pandas as pd

from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
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

# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset.csv"

OOF_FILE = (
    PROCESSED_DIR / "threshold_sensitivity_oof_predictions.csv"
)
RESULTS_FILE = (
    PROCESSED_DIR / "threshold_sensitivity_results.csv"
)
CONFUSION_FILE = (
    PROCESSED_DIR / "threshold_sensitivity_confusion_matrices.csv"
)


# --------------------------------------------------
# Model: same features and estimator as before
# --------------------------------------------------

def build_model():
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


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 70)
    print("ML CLASSIFICATION-THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values. "
            "Inspect the dataset before continuing."
        )

    y = df["market_observed"].astype(int)
    groups = df["occupation_uri"]

    if set(y.unique()) != {0, 1}:
        raise ValueError("Target must contain both classes, 0 and 1.")

    X = (
        df["occupation_label"].astype(str)
        + " "
        + df["skill_label"].astype(str)
        + " "
        + df["skill_type"].astype(str)
        + " "
        + df["relation_type"].astype(str)
    )

    # Each row receives a probability from a model that did not
    # train on any rows belonging to that row's occupation group.
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = [None] * len(df)
    fold_assignments = [None] * len(df)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Occupation overlap detected in fold {fold}."
            )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                f"Fold {fold} does not contain both target classes."
            )

        model = build_model()
        model.fit(X.iloc[train_idx], y_train)

        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(
            X.iloc[test_idx]
        )[:, positive_index]

        for idx, probability in zip(test_idx, probabilities):
            oof_probabilities[idx] = float(probability)
            fold_assignments[idx] = fold

        print(
            f"Fold {fold}/{N_SPLITS} complete | "
            f"train occupations: {len(train_groups)} | "
            f"held-out occupations: {len(test_groups)}"
        )

    if any(p is None for p in oof_probabilities):
        raise RuntimeError(
            "Some rows did not receive out-of-fold predictions."
        )

    # --------------------------------------------------
    # Save out-of-fold predictions
    # --------------------------------------------------

    oof = df[
        [
            "occupation_uri",
            "occupation_label",
            "skill_uri",
            "skill_label",
            "market_observed",
        ]
    ].copy()

    oof["fold"] = fold_assignments
    oof["predicted_probability"] = oof_probabilities

    y_true = oof["market_observed"].astype(int)
    probabilities = oof["predicted_probability"].astype(float)

    oof.to_csv(OOF_FILE, index=False)

    # Threshold-independent metrics, computed once from the
    # pooled out-of-fold probabilities.
    pooled_roc_auc = roc_auc_score(y_true, probabilities)
    pooled_pr_auc = average_precision_score(y_true, probabilities)

    print("\nPooled out-of-fold ROC-AUC:", round(pooled_roc_auc, 4))
    print("Pooled out-of-fold PR-AUC:", round(pooled_pr_auc, 4))
    print("Actual positive prevalence:", round(y_true.mean(), 4))

    # --------------------------------------------------
    # Evaluate classification thresholds
    # --------------------------------------------------

    result_rows = []
    confusion_rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        result_rows.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, predictions),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_true, predictions, zero_division=0
            ),
            "roc_auc": pooled_roc_auc,
            "pr_auc": pooled_pr_auc,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "predicted_positive_count": int(predictions.sum()),
            "actual_positive_count": int(y_true.sum()),
        })

        confusion_rows.append({
            "threshold": threshold,
            "actual_class": 0,
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        })

        confusion_rows.append({
            "threshold": threshold,
            "actual_class": 1,
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        })

    results = pd.DataFrame(result_rows)
    confusion_df = pd.DataFrame(confusion_rows)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(RESULTS_FILE, index=False)
    confusion_df.to_csv(CONFUSION_FILE, index=False)

    print("\n" + "=" * 70)
    print("THRESHOLD COMPARISON")
    print("=" * 70)

    display_columns = [
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print(OOF_FILE)
    print(RESULTS_FILE)
    print(CONFUSION_FILE)

    print(
        "\nInterpretation: class 0 means not observed in the "
        "available dataset; it does not establish that a skill "
        "is absent from an occupation."
    )


if __name__ == "__main__":
    main()
