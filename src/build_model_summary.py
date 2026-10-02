import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_RESULTS = PROCESSED_DIR / "model_results.csv"
CV_RESULTS = PROCESSED_DIR / "model_cross_validation_results.csv"
OUTPUT_FILE = PROCESSED_DIR / "model_summary.csv"


def main():

    model_results = pd.read_csv(MODEL_RESULTS)
    cv_results = pd.read_csv(CV_RESULTS)

    model = model_results[
        model_results["model"] == "TF-IDF + Logistic Regression"
    ].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "task": (
                    "Predict whether an ESCO "
                    "occupation-skill relationship "
                    "is observed in the NYC job-posting market dataset"
                ),
                "dataset_size": 2696,
                "positive_examples": 674,
                "negative_examples": 2022,
                "positive_sampling_ratio": "1:3",
                "model": "TF-IDF + Logistic Regression",
                "text_features": (
                    "occupation_label + skill_label + "
                    "skill_type + relation_type"
                ),
                "test_size": 0.20,
                "random_state": 42,
                "holdout_accuracy": model["accuracy"],
                "holdout_precision": model["precision"],
                "holdout_recall": model["recall"],
                "holdout_f1": model["f1"],
                "holdout_roc_auc": model["roc_auc"],
                "holdout_pr_auc": model["pr_auc"],
                "cv_folds": 5,
                "cv_accuracy_mean": cv_results["accuracy"].mean(),
                "cv_accuracy_std": cv_results["accuracy"].std(),
                "cv_precision_mean": cv_results["precision"].mean(),
                "cv_precision_std": cv_results["precision"].std(),
                "cv_recall_mean": cv_results["recall"].mean(),
                "cv_recall_std": cv_results["recall"].std(),
                "cv_f1_mean": cv_results["f1"].mean(),
                "cv_f1_std": cv_results["f1"].std(),
                "cv_roc_auc_mean": cv_results["roc_auc"].mean(),
                "cv_roc_auc_std": cv_results["roc_auc"].std(),
                "cv_pr_auc_mean": cv_results["pr_auc"].mean(),
                "cv_pr_auc_std": cv_results["pr_auc"].std(),
            }
        ]
    )

    summary.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("=" * 70)
    print("MODEL SUMMARY CREATED")
    print("=" * 70)

    print(summary.T.to_string(header=False))

    print()
    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()