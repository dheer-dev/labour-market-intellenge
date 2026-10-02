import pandas as pd

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "model_cross_validation_results.csv"


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MODEL CROSS-VALIDATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    # --------------------------------------------------
    # 2. Construct model text
    # --------------------------------------------------

    df["model_text"] = (
        df["occupation_label"].astype(str)
        + " "
        + df["skill_label"].astype(str)
        + " "
        + df["skill_type"].astype(str)
        + " "
        + df["relation_type"].astype(str)
    )

    X = df["model_text"]
    y = df["market_observed"]

    # --------------------------------------------------
    # 3. Define model
    # --------------------------------------------------

    model = Pipeline(
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

    # --------------------------------------------------
    # 4. Stratified 5-fold cross-validation
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
    )

    # --------------------------------------------------
    # 5. Collect fold-level results
    # --------------------------------------------------

    results = pd.DataFrame(
        {
            "fold": range(1, N_SPLITS + 1),
            "accuracy": cv_results["test_accuracy"],
            "precision": cv_results["test_precision"],
            "recall": cv_results["test_recall"],
            "f1": cv_results["test_f1"],
            "roc_auc": cv_results["test_roc_auc"],
            "pr_auc": cv_results["test_pr_auc"],
        }
    )

    # --------------------------------------------------
    # 6. Calculate mean and standard deviation
    # --------------------------------------------------

    summary = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ],
            "mean": [
                results["accuracy"].mean(),
                results["precision"].mean(),
                results["recall"].mean(),
                results["f1"].mean(),
                results["roc_auc"].mean(),
                results["pr_auc"].mean(),
            ],
            "std": [
                results["accuracy"].std(),
                results["precision"].std(),
                results["recall"].std(),
                results["f1"].std(),
                results["roc_auc"].std(),
                results["pr_auc"].std(),
            ],
        }
    )

    # --------------------------------------------------
    # 7. Print fold results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("FOLD-LEVEL RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # 8. Print summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # 9. Save results
    # --------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("CROSS-VALIDATION RESULTS SAVED")
    print("=" * 70)

    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()