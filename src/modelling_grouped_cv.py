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
)

RANDOM_STATE = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INPUT_FILE = PROCESSED_DIR / "model_dataset.csv"

FOLD_RESULTS_FILE = (
    PROCESSED_DIR / "grouped_cv_fold_results.csv"
)
SUMMARY_FILE = (
    PROCESSED_DIR / "grouped_cv_summary.csv"
)


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

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required].isnull().any().any():
        raise ValueError("Required columns contain missing values.")

    y = df["market_observed"].astype(int)
    groups = df["occupation_uri"]

    if y.nunique() != 2:
        raise ValueError("The target must contain both classes.")

    X = (
        df["occupation_label"].astype(str)
        + " "
        + df["skill_label"].astype(str)
        + " "
        + df["skill_type"].astype(str)
        + " "
        + df["relation_type"].astype(str)
    )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    rows = []

    print("=" * 70)
    print("FIVE-FOLD OCCUPATION-GROUPED CROSS-VALIDATION")
    print("=" * 70)
    print("Dataset rows:", len(df))
    print("Unique occupations:", groups.nunique())

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, groups=groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])

        if train_groups.intersection(test_groups):
            raise RuntimeError(
                f"Occupation overlap detected in fold {fold}."
            )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if y_train.nunique() != 2 or y_test.nunique() != 2:
            raise ValueError(
                f"Fold {fold} does not contain both classes "
                "in training and testing."
            )

        model = build_model()
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(X_test)[:, positive_index]

        results = {
            "fold": fold,
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_occupations": len(train_groups),
            "test_occupations": len(test_groups),
            "test_positive_count": int((y_test == 1).sum()),
            "test_negative_count": int((y_test == 0).sum()),
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "f1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(
                y_test, probabilities
            ),
        }

        rows.append(results)

        print(
            f"\nFold {fold}: "
            f"ROC-AUC={results['roc_auc']:.4f}, "
            f"PR-AUC={results['pr_auc']:.4f}, "
            f"F1={results['f1']:.4f}, "
            f"Recall={results['recall']:.4f}"
        )

    fold_results = pd.DataFrame(rows)

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
    ]

    summary = pd.DataFrame({
        "metric": metric_columns,
        "mean": [
            fold_results[c].mean() for c in metric_columns
        ],
        "std": [
            fold_results[c].std(ddof=1) for c in metric_columns
        ],
        "min": [
            fold_results[c].min() for c in metric_columns
        ],
        "max": [
            fold_results[c].max() for c in metric_columns
        ],
    })

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(FOLD_RESULTS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nSaved:")
    print(FOLD_RESULTS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
