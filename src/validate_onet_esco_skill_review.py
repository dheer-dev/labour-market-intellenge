
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "onet_esco_skill_mapping_initial_review.csv"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "onet_esco_skill_review_validation.csv"
)


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Review file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "onet_skill_label",
        "review_decision",
        "review_rationale",
        "reviewed_esco_skill_uri",
        "reviewed_esco_skill_label",
    ]

    missing = sorted(set(required_columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    valid_decisions = {
        "APPROVED",
        "REJECTED",
        "NO_DIRECT_MATCH",
        "REVIEW_REQUIRED",
    }

    invalid = set(df["review_decision"].dropna().unique()) - valid_decisions

    if invalid:
        raise ValueError(
            f"Invalid decisions: {sorted(invalid)}"
        )

    if df["onet_skill_label"].isna().any():
        raise ValueError("Missing O*NET skill labels detected.")

    if df["onet_skill_label"].duplicated().any():
        raise ValueError("Duplicate O*NET skill labels detected.")

    approved = df["review_decision"].eq("APPROVED")

    missing_uri = approved & df["reviewed_esco_skill_uri"].isna()
    missing_label = approved & df["reviewed_esco_skill_label"].isna()

    if missing_uri.any() or missing_label.any():
        raise ValueError(
            "An approved mapping has a missing ESCO URI or label."
        )

    # A mapping should only be populated when it is approved.
    not_approved = ~approved
    unexpected_mapping = (
        not_approved
        & (
            df["reviewed_esco_skill_uri"].notna()
            | df["reviewed_esco_skill_label"].notna()
        )
    )

    if unexpected_mapping.any():
        raise ValueError(
            "Unapproved rows contain reviewed ESCO mappings."
        )

    # Save a compact validation report, preserving the decisions.
    report = (
        df.groupby("review_decision", dropna=False)
        .size()
        .rename("skill_count")
        .reset_index()
    )

    report.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("O*NET–ESCO SKILL REVIEW VALIDATION")
    print("=" * 60)
    print(f"Input rows: {len(df)}")
    print(f"Unique O*NET skills: {df['onet_skill_label'].nunique()}")
    print(f"Approved mappings: {int(approved.sum())}")
    print(f"Approved mappings missing URI: {int(missing_uri.sum())}")
    print(f"Unexpected mappings on unapproved rows: {int(unexpected_mapping.sum())}")
    print(f"\nValidation report saved to:\n{OUTPUT_PATH}")
    print("\nDecision counts:")
    print(report.to_string(index=False))
    print("\nValidation passed.")


if __name__ == "__main__":
    main()
