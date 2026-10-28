
import joblib
import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MODEL_FILE = PROCESSED_DIR / "market_observation_model.joblib"
OUTPUT_FILE = PROCESSED_DIR / "model_feature_explanations.csv"

# Number of strongest features to report in each direction
TOP_N = 30


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("=" * 70)
    print("MODEL EXPLAINABILITY: FEATURE COEFFICIENT ANALYSIS")
    print("=" * 70)

    # 1. Load the previously trained pipeline
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run 'python src/modelling.py' first."
        )

    model = joblib.load(MODEL_FILE)

    # 2. Access the fitted TF-IDF vectorizer and classifier
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]

    feature_names = vectorizer.get_feature_names_out()
    classes = classifier.classes_

    if len(classes) != 2:
        raise ValueError(
            f"Expected a binary classifier, found classes: {classes}"
        )

    # For binary Logistic Regression, coef_[0] corresponds
    # to the log-odds of classes_[1] versus classes_[0].
    coefficients = classifier.coef_[0]
    positive_class = classes[1]
    negative_class = classes[0]

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature names and model coefficients do not match."
        )

    # 3. Build the complete coefficient table
    all_features = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    all_features["absolute_coefficient"] = (
        all_features["coefficient"].abs()
    )

    # Positive coefficients push the model towards classes_[1].
    # Negative coefficients push it towards classes_[0].
    all_features["direction"] = all_features["coefficient"].apply(
        lambda value: (
            "Positive"
            if value > 0
            else "Negative"
            if value < 0
            else "Neutral"
        )
    )

    all_features["associated_class"] = all_features[
        "coefficient"
    ].apply(
        lambda value: (
            str(positive_class)
            if value > 0
            else str(negative_class)
            if value < 0
            else "Neither"
        )
    )

    # 4. Select the strongest features in each direction
    positive_features = (
        all_features[all_features["coefficient"] > 0]
        .nlargest(TOP_N, "coefficient")
        .copy()
    )

    positive_features["rank_within_direction"] = range(
        1, len(positive_features) + 1
    )

    negative_features = (
        all_features[all_features["coefficient"] < 0]
        .nsmallest(TOP_N, "coefficient")
        .copy()
    )

    negative_features["rank_within_direction"] = range(
        1, len(negative_features) + 1
    )

    # 5. Combine both directions
    explanations = pd.concat(
        [positive_features, negative_features],
        ignore_index=True,
    )

    explanations["target_definition"] = (
        "Whether an ESCO occupation-skill relationship is "
        "observed in the available NYC job-posting dataset"
    )

    explanations["interpretation"] = explanations[
        "coefficient"
    ].apply(
        lambda value: (
            "Pushes prediction towards market observed"
            if value > 0
            else "Pushes prediction towards not observed"
        )
    )

    # Keep the output easy to inspect
    explanations = explanations[
        [
            "feature",
            "coefficient",
            "absolute_coefficient",
            "direction",
            "associated_class",
            "rank_within_direction",
            "interpretation",
            "target_definition",
        ]
    ]

    # 6. Save the results
    explanations.to_csv(OUTPUT_FILE, index=False)

    # 7. Report a concise summary
    print("\nClassifier classes:", list(classes))
    print("Positive class:", positive_class)
    print("Negative class:", negative_class)
    print("Total TF-IDF features:", len(feature_names))
    print("Positive features exported:", len(positive_features))
    print("Negative features exported:", len(negative_features))

    print("\nStrongest features associated with market observation:")
    print(
        positive_features[
            ["feature", "coefficient"]
        ].head(10).to_string(index=False)
    )

    print("\nStrongest features associated with non-observation:")
    print(
        negative_features[
            ["feature", "coefficient"]
        ].head(10).to_string(index=False)
    )

    print("\nExplanation file saved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 70)
    print("MODEL EXPLAINABILITY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
