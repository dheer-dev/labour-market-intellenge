
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "onet_esco_skill_mapping_review_template.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "onet_esco_skill_mapping_initial_review.csv"
)


# ---------------------------------------------------------
# INITIAL REVIEW DECISIONS
# ---------------------------------------------------------
# These are preliminary decisions based on the candidate
# labels currently available, not a validated crosswalk.
#
# APPROVED:
#   Exact label candidate, still requiring concept-level
#   verification before final research approval.
#
# REVIEW_REQUIRED:
#   A potentially related candidate needs manual review.
#
# NO_DIRECT_MATCH:
#   No convincing match among the displayed top candidates.
#   This does NOT mean no equivalent exists anywhere in ESCO.
# ---------------------------------------------------------

REVIEW_RULES = {
    "Mathematics": (
        "APPROVED",
        (
            "Exact label match with ESCO Mathematics. "
            "Provisional approval only; verify the O*NET and "
            "ESCO concept definitions before final approval."
        ),
    ),

    "Learning Strategies": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidate 'use learning strategies' appears "
            "potentially related. Verify the concept definitions."
        ),
    ),

    "Programming": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidates include web programming and "
            "computer programming. Determine whether either "
            "matches the scope of the O*NET element."
        ),
    ),

    "Equipment Maintenance": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidates include ensuring or advising on "
            "equipment maintenance. Verify semantic equivalence."
        ),
    ),

    "Quality Control Analysis": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidate 'conduct quality control analysis' "
            "appears potentially related. Verify the definitions."
        ),
    ),

    "Systems Analysis": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidate 'perform system analysis' may be "
            "related. Confirm the scope and meaning."
        ),
    ),

    "Troubleshooting": (
        "REVIEW_REQUIRED",
        (
            "ESCO includes the candidate 'troubleshoot'. "
            "Verify whether it represents the same skill concept."
        ),
    ),

    "Service Orientation": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidate 'ensure client orientation' may be "
            "related, but the meaning may differ. Verify definitions."
        ),
    ),

    "Complex Problem Solving": (
        "REVIEW_REQUIRED",
        (
            "The displayed candidates do not establish equivalence. "
            "Inspect the full ESCO skill catalogue and definitions."
        ),
    ),

    "Critical Thinking": (
        "REVIEW_REQUIRED",
        (
            "The displayed candidates do not establish equivalence. "
            "Search the full ESCO catalogue and compare definitions."
        ),
    ),

    "Judgment and Decision Making": (
        "REVIEW_REQUIRED",
        (
            "Decision-making candidates may be related, but the "
            "scope of the O*NET element must be checked."
        ),
    ),

    "Management of Financial Resources": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidates involving control or management of "
            "financial resources may be related. Verify definitions."
        ),
    ),

    "Management of Material Resources": (
        "REVIEW_REQUIRED",
        (
            "ESCO candidate labels involving material resources "
            "may be related. Verify the underlying concepts."
        ),
    ),
}


DEFAULT_DECISION = "NO_DIRECT_MATCH"

DEFAULT_RATIONALE = (
    "No convincing equivalent was identified among the displayed "
    "top candidates using label similarity alone. This is not proof "
    "that no equivalent exists in the full ESCO skill catalogue."
)


VALID_DECISIONS = {
    "APPROVED",
    "REJECTED",
    "NO_DIRECT_MATCH",
    "REVIEW_REQUIRED",
}


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Review template not found: {INPUT_PATH}\n"
            "Run src/build_onet_esco_skill_review_template.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "onet_skill_label",
        "candidate_1_esco_uri",
        "candidate_1_esco_label",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "The review template is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # Work with a copy so the original review template remains unchanged.
    df = df.copy()

    # Initialise the output columns explicitly as strings.
    # This fixes the pandas float64 assignment error when the
    # columns initially contain only missing values.
    string_columns = [
        "review_decision",
        "review_rationale",
        "reviewed_esco_skill_uri",
        "reviewed_esco_skill_label",
    ]

    for column in string_columns:
        if column not in df.columns:
            df[column] = pd.Series(
                pd.NA,
                index=df.index,
                dtype="string",
            )
        else:
            df[column] = df[column].astype("string")

    # Ensure candidate fields are treated as strings as well.
    df["candidate_1_esco_uri"] = (
        df["candidate_1_esco_uri"].astype("string")
    )
    df["candidate_1_esco_label"] = (
        df["candidate_1_esco_label"].astype("string")
    )

    # -----------------------------------------------------
    # APPLY PRELIMINARY DECISIONS
    # -----------------------------------------------------

    for index, row in df.iterrows():
        skill_label = str(row["onet_skill_label"]).strip()

        decision, rationale = REVIEW_RULES.get(
            skill_label,
            (DEFAULT_DECISION, DEFAULT_RATIONALE),
        )

        df.at[index, "review_decision"] = decision
        df.at[index, "review_rationale"] = rationale

    # -----------------------------------------------------
    # POPULATE REVIEWED ESCO FIELDS FOR MATHEMATICS ONLY
    # -----------------------------------------------------
    # This remains a provisional mapping. The other candidates
    # are not automatically accepted based on fuzzy similarity.
    # -----------------------------------------------------

    math_mask = (
        df["onet_skill_label"]
        .str.strip()
        .str.casefold()
        .eq("mathematics")
    )

    if math_mask.sum() != 1:
        raise ValueError(
            "Expected exactly one Mathematics row in the review "
            f"template; found {int(math_mask.sum())}."
        )

    if df.loc[math_mask, "candidate_1_esco_label"].isna().any():
        raise ValueError(
            "The Mathematics row has no first ESCO candidate label."
        )

    exact_math_mask = (
        df["candidate_1_esco_label"]
        .str.strip()
        .str.casefold()
        .eq("mathematics")
    )

    if not exact_math_mask.loc[math_mask].all():
        raise ValueError(
            "The first ESCO candidate for Mathematics is not an "
            "exact normalized label match. Inspect the candidates "
            "before creating the provisional mapping."
        )

    df.loc[math_mask, "reviewed_esco_skill_uri"] = (
        df.loc[math_mask, "candidate_1_esco_uri"].to_numpy()
    )

    df.loc[math_mask, "reviewed_esco_skill_label"] = (
        df.loc[math_mask, "candidate_1_esco_label"].to_numpy()
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    invalid_decisions = (
        set(df["review_decision"].dropna().unique())
        - VALID_DECISIONS
    )

    if invalid_decisions:
        raise ValueError(
            f"Invalid review decisions found: {invalid_decisions}"
        )

    if df["onet_skill_label"].isna().any():
        raise ValueError(
            "Some rows have missing O*NET skill labels."
        )

    if df["onet_skill_label"].duplicated().any():
        duplicates = df.loc[
            df["onet_skill_label"].duplicated(keep=False),
            "onet_skill_label",
        ].tolist()

        raise ValueError(
            "Duplicate O*NET skill labels found in the review "
            f"template: {duplicates}"
        )

    approved_mask = df["review_decision"].eq("APPROVED")

    approved_without_uri = (
        approved_mask
        & df["reviewed_esco_skill_uri"].isna()
    )

    if approved_without_uri.any():
        missing_skills = df.loc[
            approved_without_uri, "onet_skill_label"
        ].tolist()

        raise ValueError(
            "Approved rows are missing reviewed ESCO URIs: "
            f"{missing_skills}"
        )

    # Do not leave reviewed mappings on unapproved rows.
    unapproved_mask = ~approved_mask

    df.loc[
        unapproved_mask,
        ["reviewed_esco_skill_uri", "reviewed_esco_skill_label"],
    ] = pd.NA

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(OUTPUT_PATH, index=False)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    print("=" * 65)
    print("O*NET–ESCO INITIAL SKILL REVIEW")
    print("=" * 65)

    print(f"\nInput:  {INPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"\nReview rows: {len(df)}")
    print(
        "Unique O*NET skills:",
        df["onet_skill_label"].nunique(),
    )

    print("\nReview decisions:")
    print(
        df["review_decision"]
        .value_counts()
        .reindex(
            [
                "APPROVED",
                "REVIEW_REQUIRED",
                "NO_DIRECT_MATCH",
                "REJECTED",
            ],
            fill_value=0,
        )
        .to_string()
    )

    print("\nProvisional approved mappings:")
    print(
        df.loc[
            approved_mask,
            [
                "onet_skill_label",
                "reviewed_esco_skill_label",
                "review_decision",
            ],
        ].to_string(index=False)
    )

    print(
        "\nApproved rows missing reviewed URI:",
        int(approved_without_uri.sum()),
    )

    print("\nValidation passed.")
    print(
        "Important: these are preliminary review decisions, "
        "not a validated semantic crosswalk."
    )


if __name__ == "__main__":
    main()
