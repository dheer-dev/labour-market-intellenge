import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
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

RESULTS_FILE = PROCESSED_DIR / "grouped_model_results.csv"
PREDICTIONS_FILE = (
    PROCESSED_DIR / "grouped_model_test_predictions.csv"
)
SPLIT_FILE = PROCESSED_DIR / "grouped_model_split_summary.csv"
CONFUSION_FILE = (
    PROCESSED_DIR / "grouped_model_confusion_matrix.csv"
)
MODEL_FILE = (
    PROCESSED_DIR / "market_observation_model_grouped.joblib"
)


# --------------------------------------------------
# Evaluation helper
# --------------------------------------------------

def evaluate_model(name, y_true, y_pred, y_probability):
    return {
        "model": name,
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
        "roc_auc": roc_auc_score(
            y_true, y_probability
        ),
        "pr_auc": average_precision_score(
            y_true, y_probability
        ),
    }


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("OCCUPATION-GROUPED MODEL EVALUATION")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "occupation_uri",
        "occupation_label",
        "skill_label",
        "skill_type",
        "relation_type",
        "market_observed",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values. "
            "Inspect the dataset before continuing."
        )

    df["market_observed"] = df["market_observed"].astype(int)

    if set(df["market_observed"].unique()) != {0, 1}:
        raise ValueError(
            "The target must contain both classes: 0 and 1."
        )

    # Use exactly the same text representation as the original model.
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
    groups = df["occupation_uri"]

    # Hold out entire occupations rather than individual rows.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])

    # Verify the split genuinely holds out occupations.
    overlap = train_groups.intersection(test_groups)

    if overlap:
        raise RuntimeError(
            f"Occupation leakage detected: {len(overlap)} "
            "occupations appear in both sets."
        )

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "This grouped split does not contain both target "
            "classes in training and testing. Do not interpret "
            "the metrics. We need to inspect the group distribution "
            "and choose a defensible alternative split."
        )

    print("\nDataset rows:", len(df))
    print("Unique occupations:", groups.nunique())
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Training occupations:", len(train_groups))
    print("Testing occupations:", len(test_groups))
    print("Overlapping occupations:", len(overlap))

    print("\nTraining target distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting target distribution:")
    print(y_test.value_counts().sort_index())

    # --------------------------------------------------
    # 1. Majority-class baseline
    # --------------------------------------------------

    majority_class = int(y_train.mode().iloc[0])

    baseline_predictions = [
        majority_class
    ] * len(y_test)

    baseline_probabilities = [
        float(majority_class)
    ] * len(y_test)

    baseline_results = evaluate_model(
        "Majority Class Baseline (Grouped Split)",
        y_test,
        baseline_predictions,
        baseline_probabilities,
    )

    # --------------------------------------------------
    # 2. Same TF-IDF + Logistic Regression model
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

    predictions = model.predict(X_test)

    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[
        :, positive_class_index
    ]

    model_results = evaluate_model(
        "TF-IDF + Logistic Regression (Grouped Split)",
        y_test,
        predictions,
        probabilities,
    )

    # --------------------------------------------------
    # 3. Save separate experiment outputs
    # --------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [baseline_results, model_results]
    )
    results.to_csv(RESULTS_FILE, index=False)

    test_output = test_df[
        [
            "occupation_uri",
            "occupation_label",
            "skill_uri",
            "skill_label",
            "market_observed",
        ]
    ].copy()

    test_output["predicted_market_observed"] = predictions
    test_output["predicted_probability"] = probabilities
    test_output.to_csv(PREDICTIONS_FILE, index=False)

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "row_count": len(train_df),
                "occupation_count": len(train_groups),
                "observed_count": int((y_train == 1).sum()),
                "non_observed_count": int((y_train == 0).sum()),
            },
            {
                "split": "test",
                "row_count": len(test_df),
                "occupation_count": len(test_groups),
                "observed_count": int((y_test == 1).sum()),
                "non_observed_count": int((y_test == 0).sum()),
            },
        ]
    )
    split_summary.to_csv(SPLIT_FILE, index=False)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    confusion_df.to_csv(CONFUSION_FILE)

    # Save this experiment's model separately.
    joblib.dump(model, MODEL_FILE)

    # --------------------------------------------------
    # 4. Report results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("GROUPED-SPLIT EVALUATION RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_df)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=[
                "Not observed in available market data",
                "Observed in available market data",
            ],
            zero_division=0,
        )
    )

    print("\nSaved outputs:")
    print(RESULTS_FILE)
    print(PREDICTIONS_FILE)
    print(SPLIT_FILE)
    print(CONFUSION_FILE)
    print(MODEL_FILE)

    print("\nInterpretation reminder:")
    print(
        "Class 0 means not observed in the available dataset; "
        "it does not prove that the skill is genuinely absent "
        "from the occupation."
    )


if __name__ == "__main__":
    main()
