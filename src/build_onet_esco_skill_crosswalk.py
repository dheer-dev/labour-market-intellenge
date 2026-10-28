
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

ONET_PATH = PROCESSED / "onet_target_occupation_skills.csv"
REVIEW_PATH = PROCESSED / "onet_esco_skill_mapping_initial_review.csv"
ESCO_PATH = PROCESSED / "esco_occupation_skill_relations.csv"

CROSSWALK_PATH = PROCESSED / "onet_esco_skill_crosswalk.csv"
OCCUPATION_SKILLS_PATH = PROCESSED / "onet_target_skills_with_crosswalk.csv"
SUMMARY_PATH = PROCESSED / "onet_esco_crosswalk_validation.csv"


def require_columns(df, required, filename):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")


def main():
    for path in [ONET_PATH, REVIEW_PATH, ESCO_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    onet = pd.read_csv(ONET_PATH)
    review = pd.read_csv(REVIEW_PATH)
    esco = pd.read_csv(ESCO_PATH)

    require_columns(
        onet,
        [
            "target_occupation",
            "onet_soc_code",
            "onet_occupation_title",
            "element_id",
            "element_name",
            "onet_skill_category",
            "onet_importance",
            "onet_level",
            "source_dataset",
        ],
        ONET_PATH.name,
    )

    require_columns(
        review,
        [
            "onet_element_id",
            "onet_skill_label",
            "onet_skill_category",
            "reviewed_esco_skill_uri",
            "reviewed_esco_skill_label",
            "review_decision",
            "review_rationale",
        ],
        REVIEW_PATH.name,
    )

    require_columns(
        esco,
        [
            "occupation_uri",
            "occupation_label",
            "relation_type",
            "skill_type",
            "skill_uri",
            "skill_label",
        ],
        ESCO_PATH.name,
    )

    # One review record per O*NET element.
    if review["onet_element_id"].duplicated().any():
        raise ValueError(
            "The review file contains duplicate O*NET element IDs."
        )

    # Keep O*NET element IDs as strings for a reliable join.
    onet["element_id"] = onet["element_id"].astype("string").str.strip()
    review["onet_element_id"] = (
        review["onet_element_id"].astype("string").str.strip()
    )

    # Build a compact crosswalk from the actual reviewed decisions.
    crosswalk_columns = [
        "onet_element_id",
        "onet_skill_label",
        "onet_skill_category",
        "review_decision",
        "review_rationale",
        "reviewed_esco_skill_uri",
        "reviewed_esco_skill_label",
        "candidate_1_esco_uri",
        "candidate_1_esco_label",
        "candidate_1_string_similarity",
        "candidate_1_token_overlap",
        "candidate_2_esco_uri",
        "candidate_2_esco_label",
        "candidate_3_esco_uri",
        "candidate_3_esco_label",
    ]

    crosswalk_columns = [
        column for column in crosswalk_columns if column in review.columns
    ]

    crosswalk = review[crosswalk_columns].copy()

    # A mapping is considered usable only if it was approved and has a URI.
    approved = crosswalk["review_decision"].eq("APPROVED")
    has_uri = crosswalk["reviewed_esco_skill_uri"].notna()

    crosswalk["mapping_usable"] = approved & has_uri

    # Check whether each reviewed ESCO skill URI appears in the ESCO
    # occupation-skill relation dataset. This is a coverage check,
    # not proof of semantic equivalence.
    esco_uris = set(esco["skill_uri"].dropna().astype(str))

    crosswalk["reviewed_skill_present_in_esco_relations"] = (
        crosswalk["reviewed_esco_skill_uri"]
        .astype("string")
        .isin(esco_uris)
        & crosswalk["mapping_usable"]
    )

    crosswalk["crosswalk_status"] = crosswalk["review_decision"].astype(
        "string"
    )

    crosswalk.loc[
        crosswalk["mapping_usable"]
        & ~crosswalk["reviewed_skill_present_in_esco_relations"],
        "crosswalk_status",
    ] = "APPROVED_BUT_URI_NOT_FOUND_IN_RELATIONS"

    # Preserve all 35 reviewed skills in the crosswalk.
    if crosswalk["onet_element_id"].nunique() != len(crosswalk):
        raise ValueError("Crosswalk contains duplicate O*NET element IDs.")

    # Attach reviewed skill mappings to occupation-specific O*NET records.
    occupation_skills = onet.merge(
        crosswalk,
        left_on="element_id",
        right_on="onet_element_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched = occupation_skills["_merge"].ne("both")
    if unmatched.any():
        labels = occupation_skills.loc[
            unmatched, ["element_id", "element_name"]
        ].drop_duplicates()

        raise ValueError(
            "Some O*NET skill records have no review decision:\n"
            + labels.to_string(index=False)
        )

    occupation_skills = occupation_skills.drop(columns=["_merge"])

    # Validate approved mappings.
    approved_rows = crosswalk["review_decision"].eq("APPROVED")
    approved_missing_uri = (
        approved_rows & crosswalk["reviewed_esco_skill_uri"].isna()
    )

    if approved_missing_uri.any():
        raise ValueError("An approved mapping is missing its ESCO URI.")

    # Save outputs.
    crosswalk.to_csv(CROSSWALK_PATH, index=False)
    occupation_skills.to_csv(OCCUPATION_SKILLS_PATH, index=False)

    # Produce a concise validation summary.
    decision_counts = (
        crosswalk["review_decision"]
        .value_counts(dropna=False)
        .to_dict()
    )

    summary_rows = [
        ("reviewed_skill_elements", len(crosswalk)),
        ("unique_reviewed_skill_elements", crosswalk["onet_element_id"].nunique()),
        ("occupation_skill_records", len(occupation_skills)),
        ("occupations_in_onet_dataset", occupation_skills["target_occupation"].nunique()),
        ("approved_mappings", int(approved_rows.sum())),
        ("approved_mappings_missing_uri", int(approved_missing_uri.sum())),
        (
            "approved_uris_present_in_esco_relations",
            int(crosswalk["reviewed_skill_present_in_esco_relations"].sum()),
        ),
        ("review_required", int(decision_counts.get("REVIEW_REQUIRED", 0))),
        ("no_direct_match", int(decision_counts.get("NO_DIRECT_MATCH", 0))),
        ("rejected", int(decision_counts.get("REJECTED", 0))),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])
    summary.to_csv(SUMMARY_PATH, index=False)

    print("=" * 65)
    print("O*NET–ESCO SKILL CROSSWALK BUILD")
    print("=" * 65)
    print(f"Crosswalk: {CROSSWALK_PATH}")
    print(f"Occupation-level skills: {OCCUPATION_SKILLS_PATH}")
    print(f"Validation summary: {SUMMARY_PATH}")

    print("\nReview decisions:")
    print(crosswalk["review_decision"].value_counts(dropna=False).to_string())

    print("\nOccupation-level records:")
    print(
        occupation_skills.groupby("target_occupation")
        .size()
        .to_string()
    )

    print("\nValidation summary:")
    print(summary.to_string(index=False))

    print(
        "\nImportant: O*NET importance and level remain separate from "
        "ESCO relationships and NYC job-posting prevalence."
    )
    print(
        "A URI appearing in the ESCO relation file confirms dataset "
        "presence only; it does not independently validate a skill match."
    )
    print("\nCrosswalk build completed successfully.")


if __name__ == "__main__":
    main()
