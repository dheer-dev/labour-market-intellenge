import pandas as pd
from pathlib import Path


USER_FILE = Path(
    "data/processed/user_skill_representation.csv"
)

COMPARISON_FILE = Path(
    "data/processed/user_occupation_skill_comparison.csv"
)


TARGET_OCCUPATIONS = {
    "civil engineer",
    "construction manager",
    "environmental engineer",
    "lawyer",
    "project manager",
}


user_df = pd.read_csv(USER_FILE)
comparison_df = pd.read_csv(COMPARISON_FILE)


print("=" * 70)
print("PHASE 6.4 — VALIDATE USER–OCCUPATION SKILL COMPARISON")
print("=" * 70)

print(
    f"\nUser representation shape: "
    f"{user_df.shape}"
)

print(
    f"Comparison shape: "
    f"{comparison_df.shape}"
)


# ---------------------------------------------------------------
# 1. Required columns
# ---------------------------------------------------------------

required_columns = [
    "user_id",
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "user_skill_status",
    "personal_gap",
    "gap_evidence",
    "market_observed",
    "market_evidence",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
    "skill_rank",
    "demand_category",
]


for column in required_columns:

    if column not in comparison_df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


print("\n[PASS] All required columns are present.")


# ---------------------------------------------------------------
# 2. No empty comparison
# ---------------------------------------------------------------

if comparison_df.empty:
    raise ValueError(
        "Comparison dataset is empty."
    )


print("[PASS] Comparison dataset is not empty.")


# ---------------------------------------------------------------
# 3. Validate target occupations
# ---------------------------------------------------------------

actual_occupations = set(
    comparison_df["occupation_label"]
    .str.lower()
    .unique()
)

if actual_occupations != TARGET_OCCUPATIONS:

    missing = TARGET_OCCUPATIONS - actual_occupations
    unexpected = actual_occupations - TARGET_OCCUPATIONS

    raise ValueError(
        f"Occupation mismatch. "
        f"Missing={missing}, "
        f"Unexpected={unexpected}"
    )


print(
    "[PASS] All five target occupations are present."
)


# ---------------------------------------------------------------
# 4. No duplicate occupation-skill relations
# ---------------------------------------------------------------

duplicates = comparison_df.duplicated(
    subset=[
        "occupation_uri",
        "skill_uri",
    ]
)

if duplicates.any():
    raise ValueError(
        "Duplicate occupation-skill relations found."
    )


print(
    "[PASS] No duplicate occupation-skill relations."
)


# ---------------------------------------------------------------
# 5. Validate user skill URIs
# ---------------------------------------------------------------

user_skill_uris = set(
    user_df["esco_skill_uri"]
)

possessed_rows = comparison_df[
    comparison_df["user_skill_status"]
    == "Possessed"
]

possessed_uris = set(
    possessed_rows["skill_uri"]
)

if not possessed_uris.issubset(user_skill_uris):

    raise ValueError(
        "A possessed skill does not exist in "
        "the user's canonical ESCO skill representation."
    )


print(
    "[PASS] All possessed skills come from the "
    "canonical user skill representation."
)


# ---------------------------------------------------------------
# 6. Validate status ↔ personal_gap
# ---------------------------------------------------------------

expected_gap = (
    comparison_df["user_skill_status"]
    == "Gap"
)

if not (
    comparison_df["personal_gap"]
    == expected_gap
).all():

    raise ValueError(
        "personal_gap does not consistently match "
        "user_skill_status."
    )


print(
    "[PASS] Personal-gap flag is consistent "
    "with user skill status."
)


# ---------------------------------------------------------------
# 7. Validate binary gap logic
# ---------------------------------------------------------------

valid_statuses = {
    "Possessed",
    "Gap",
}

actual_statuses = set(
    comparison_df["user_skill_status"]
)

if not actual_statuses.issubset(valid_statuses):

    raise ValueError(
        f"Unexpected user skill status values: "
        f"{actual_statuses - valid_statuses}"
    )


print(
    "[PASS] User skill status contains only "
    "'Possessed' and 'Gap'."
)


# ---------------------------------------------------------------
# 8. Validate market-observed flag
# ---------------------------------------------------------------

expected_market_observed = (
    comparison_df["skill_posting_count"]
    .notna()
)

if not (
    comparison_df["market_observed"]
    == expected_market_observed
).all():

    raise ValueError(
        "market_observed is inconsistent with "
        "market posting evidence."
    )


print(
    "[PASS] Market-observed flag is consistent "
    "with market evidence."
)


# ---------------------------------------------------------------
# 9. Validate market evidence text
# ---------------------------------------------------------------

for _, row in comparison_df.iterrows():

    if row["market_observed"]:

        if row["market_evidence"] != (
            "Observed in market postings"
        ):
            raise ValueError(
                "Incorrect market evidence label "
                "for observed skill."
            )

    else:

        if row["market_evidence"] != (
            "Not observed in current market dataset"
        ):
            raise ValueError(
                "Incorrect market evidence label "
                "for unobserved skill."
            )


print(
    "[PASS] Market evidence labels are consistent."
)


# ---------------------------------------------------------------
# 10. Validate user IDs
# ---------------------------------------------------------------

user_ids = set(
    user_df["user_id"]
)

comparison_user_ids = set(
    comparison_df["user_id"]
)

if comparison_user_ids != user_ids:

    raise ValueError(
        "Comparison contains unexpected user IDs."
    )


print(
    "[PASS] User ID is consistent."
)


# ---------------------------------------------------------------
# 11. Summary
# ---------------------------------------------------------------

print("\nValidated comparison summary:")

for occupation in sorted(TARGET_OCCUPATIONS):

    rows = comparison_df[
        comparison_df["occupation_label"]
        .str.lower()
        == occupation
    ]

    possessed = (
        rows["user_skill_status"]
        == "Possessed"
    ).sum()

    gaps = (
        rows["user_skill_status"]
        == "Gap"
    ).sum()

    market_observed = (
        rows["market_observed"]
    ).sum()

    print(
        f"  {occupation}: "
        f"{len(rows)} ESCO skills | "
        f"{possessed} possessed | "
        f"{gaps} gaps | "
        f"{market_observed} market-observed"
    )


print(
    "\nPHASE 6.4 VALIDATION: ALL CHECKS PASSED"
)