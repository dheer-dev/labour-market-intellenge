import pandas as pd

from pathlib import Path

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_FILE = (
    PROCESSED_DIR / "threshold_sensitivity_oof_predictions.csv"
)

ERRORS_FILE = (
    PROCESSED_DIR / "model_error_details.csv"
)
SUMMARY_FILE = (
    PROCESSED_DIR / "model_error_summary_by_threshold.csv"
)
OCCUPATION_FILE = (
    PROCESSED_DIR / "model_errors_by_occupation.csv"
)
SKILL_FILE = (
    PROCESSED_DIR / "model_errors_by_skill.csv"
)

THRESHOLDS = [0.30, 0.50]


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print("=" * 70)
    print("ML MODEL ERROR ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "occupation_uri",
        "occupation_label",
        "skill_uri",
        "skill_label",
        "market_observed",
        "fold",
        "predicted_probability",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Required columns contain missing values."
        )

    df["market_observed"] = df["market_observed"].astype(int)

    if not set(df["market_observed"].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")

    if not df["predicted_probability"].between(0, 1).all():
        raise ValueError(
            "Predicted probabilities must be between 0 and 1."
        )

    print("Dataset rows:", len(df))
    print("Unique occupations:", df["occupation_uri"].nunique())
    print("Unique skills:", df["skill_uri"].nunique())

    # --------------------------------------------------
    # 1. Label every row at each threshold
    # --------------------------------------------------

    all_errors = []
    threshold_summaries = []
    occupation_summaries = []
    skill_summaries = []

    for threshold in THRESHOLDS:
        part = df.copy()

        part["threshold"] = threshold
        part["predicted_label"] = (
            part["predicted_probability"] >= threshold
        ).astype(int)

        part["prediction_status"] = "True Negative"

        part.loc[
            (part["market_observed"] == 1)
            & (part["predicted_label"] == 1),
            "prediction_status",
        ] = "True Positive"

        part.loc[
            (part["market_observed"] == 0)
            & (part["predicted_label"] == 0),
            "prediction_status",
        ] = "True Negative"

        part.loc[
            (part["market_observed"] == 0)
            & (part["predicted_label"] == 1),
            "prediction_status",
        ] = "False Positive"

        part.loc[
            (part["market_observed"] == 1)
            & (part["predicted_label"] == 0),
            "prediction_status",
        ] = "False Negative"

        part["is_error"] = part["prediction_status"].isin(
            ["False Positive", "False Negative"]
        )

        part["error_type"] = part["prediction_status"].where(
            part["is_error"],
            "Correct Prediction",
        )

        # Overall metrics at this threshold.
        tp = int((part["prediction_status"] == "True Positive").sum())
        tn = int((part["prediction_status"] == "True Negative").sum())
        fp = int((part["prediction_status"] == "False Positive").sum())
        fn = int((part["prediction_status"] == "False Negative").sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        threshold_summaries.append({
            "threshold": threshold,
            "total_rows": len(part),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "total_errors": fp + fn,
            "error_rate": (fp + fn) / len(part),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        all_errors.append(part)

        # --------------------------------------------------
        # 2. Summarise errors by occupation
        # --------------------------------------------------

        grouped = part.groupby(
            ["occupation_uri", "occupation_label"],
            dropna=False,
        )

        occ = grouped.agg(
            row_count=("market_observed", "size"),
            actual_observed=("market_observed", "sum"),
            predicted_observed=("predicted_label", "sum"),
            true_positives=(
                "prediction_status",
                lambda s: (s == "True Positive").sum(),
            ),
            true_negatives=(
                "prediction_status",
                lambda s: (s == "True Negative").sum(),
            ),
            false_positives=(
                "prediction_status",
                lambda s: (s == "False Positive").sum(),
            ),
            false_negatives=(
                "prediction_status",
                lambda s: (s == "False Negative").sum(),
            ),
        ).reset_index()

        occ["threshold"] = threshold
        occ["total_errors"] = (
            occ["false_positives"] + occ["false_negatives"]
        )
        occ["error_rate"] = occ["total_errors"] / occ["row_count"]

        occupation_summaries.append(occ)

        # --------------------------------------------------
        # 3. Summarise errors by skill label
        # --------------------------------------------------

        skill_grouped = part.groupby(
            ["skill_uri", "skill_label"],
            dropna=False,
        )

        skill = skill_grouped.agg(
            row_count=("market_observed", "size"),
            actual_observed=("market_observed", "sum"),
            predicted_observed=("predicted_label", "sum"),
            false_positives=(
                "prediction_status",
                lambda s: (s == "False Positive").sum(),
            ),
            false_negatives=(
                "prediction_status",
                lambda s: (s == "False Negative").sum(),
            ),
        ).reset_index()

        skill["threshold"] = threshold
        skill["total_errors"] = (
            skill["false_positives"] + skill["false_negatives"]
        )
        skill["error_rate"] = skill["total_errors"] / skill["row_count"]

        skill_summaries.append(skill)

    # --------------------------------------------------
    # 4. Combine and save outputs
    # --------------------------------------------------

    errors_df = pd.concat(all_errors, ignore_index=True)
    threshold_df = pd.DataFrame(threshold_summaries)
    occupation_df = pd.concat(
        occupation_summaries, ignore_index=True
    )
    skill_df = pd.concat(skill_summaries, ignore_index=True)

    # Detailed file contains every row at both thresholds.
    errors_df.to_csv(ERRORS_FILE, index=False)
    threshold_df.to_csv(SUMMARY_FILE, index=False)
    occupation_df.to_csv(OCCUPATION_FILE, index=False)
    skill_df.to_csv(SKILL_FILE, index=False)

    # --------------------------------------------------
    # 5. Print findings
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL ERROR SUMMARY")
    print("=" * 70)

    print(
        threshold_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    for threshold in THRESHOLDS:
        print("\n" + "-" * 70)
        print(f"THRESHOLD: {threshold:.2f}")
        print("-" * 70)

        subset = errors_df[
            errors_df["threshold"] == threshold
        ]

        print("\nPrediction status counts:")
        print(subset["prediction_status"].value_counts())

        print("\nFalse positives with highest predicted probabilities:")
        fp = subset[
            subset["prediction_status"] == "False Positive"
        ].sort_values(
            "predicted_probability", ascending=False
        )

        if fp.empty:
            print("No false positives.")
        else:
            print(
                fp[
                    [
                        "occupation_label",
                        "skill_label",
                        "market_observed",
                        "predicted_probability",
                        "fold",
                    ]
                ].head(10).to_string(index=False)
            )

        print("\nFalse negatives with lowest predicted probabilities:")
        fn = subset[
            subset["prediction_status"] == "False Negative"
        ].sort_values(
            "predicted_probability", ascending=True
        )

        if fn.empty:
            print("No false negatives.")
        else:
            print(
                fn[
                    [
                        "occupation_label",
                        "skill_label",
                        "market_observed",
                        "predicted_probability",
                        "fold",
                    ]
                ].head(10).to_string(index=False)
            )

        occ = occupation_df[
            occupation_df["threshold"] == threshold
        ].copy()

        # Show only occupations with at least three evaluated rows,
        # so single-row groups do not dominate the ranking.
        eligible = occ[occ["row_count"] >= 3]

        print("\nOccupations with most errors (minimum 3 rows):")
        if eligible.empty:
            print("No occupations meet the minimum row count.")
        else:
            print(
                eligible.sort_values(
                    ["total_errors", "row_count"],
                    ascending=[False, False],
                )[
                    [
                        "occupation_label",
                        "row_count",
                        "actual_observed",
                        "predicted_observed",
                        "false_positives",
                        "false_negatives",
                        "total_errors",
                        "error_rate",
                    ]
                ].head(10).to_string(
                    index=False,
                    float_format=lambda x: f"{x:.3f}",
                )
            )

    print("\n" + "=" * 70)
    print("SAVED OUTPUTS")
    print("=" * 70)
    print(ERRORS_FILE)
    print(SUMMARY_FILE)
    print(OCCUPATION_FILE)
    print(SKILL_FILE)

    print(
        "\nInterpretation reminder: 'not observed' means the "
        "relationship was not labelled as observed in this dataset. "
        "It does not prove that the skill is absent from the occupation."
    )


if __name__ == "__main__":
    main()
