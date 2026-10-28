
import pandas as pd
import joblib

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

RANDOM_STATE = 42
TEST_SIZE = 0.20

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "model_dataset.csv"
RESULTS_FILE = PROCESSED_DIR / "model_results.csv"
MODEL_FILE = PROCESSED_DIR / "market_observation_model.joblib"


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred, y_probability):

    results = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "pr_auc": average_precision_score(
            y_true,
            y_probability,
        ),
    }

    return results


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    print("Dataset shape:", df.shape)

    required_columns = [
        "occupation_label",
        "skill_label",
        "skill_type",
        "relation_type",
        "market_observed",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------
    # 2. Construct text representation
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
    # 3. Stratified train/test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print()
    print("Training class distribution:")
    print(y_train.value_counts())

    print()
    print("Testing class distribution:")
    print(y_test.value_counts())

    # --------------------------------------------------
    # 4. Majority-class baseline
    # --------------------------------------------------

    majority_class = y_train.mode()[0]

    baseline_predictions = [majority_class] * len(y_test)

    # Majority baseline assigns zero probability to the
    # positive class when the majority class is 0.
    baseline_probability = [
        1.0 if majority_class == 1 else 0.0
    ] * len(y_test)

    baseline_results = evaluate_model(
        "Majority Class Baseline",
        y_test,
        baseline_predictions,
        baseline_probability,
    )

    # --------------------------------------------------
    # 5. TF-IDF + Logistic Regression
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

    model.fit(X_train, y_train)

    # Save the fitted pipeline, including its fitted
    # TF-IDF vocabulary and Logistic Regression model.
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print()
    print("Trained model saved:", MODEL_FILE)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    model_results = evaluate_model(
        "TF-IDF + Logistic Regression",
        y_test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------
    # 6. Print results
    # --------------------------------------------------

    results = pd.DataFrame(
        [
            baseline_results,
            model_results,
        ]
    )

    print()
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    print(
        results[
            [
                "model",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # 7. Confusion matrix
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("LOGISTIC REGRESSION CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print(cm)

    print()
    print("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Not observed",
                "Market observed",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # 8. Save evaluation results
    # --------------------------------------------------

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    print("=" * 70)
    print("MODEL RESULTS SAVED")
    print("=" * 70)

    print("Results:", RESULTS_FILE)
    print("Model:", MODEL_FILE)


if __name__ == "__main__":
    main()
