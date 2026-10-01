import pandas as pd
from pathlib import Path


COMPARISON_FILE = Path(
    "data/processed/user_occupation_skill_comparison.csv"
)

GAP_FILE = Path(
    "data/processed/personal_skill_gaps.csv"
)


comparison_df = pd.read_csv(COMPARISON_FILE)
gap_df = pd.read_csv(GAP_FILE)


print("=" * 70)
print("PHASE 6.5 — VALIDATE MARKET-RELEVANT PERSONAL SKILL GAPS")
print("=" * 70)

print(
    f"\nComparison shape: "
    f"{comparison_df.shape}"
)

print(
    f"Personal gap shape: "
    f"{gap_df.shape}"
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
    "gap_type",
    "evidence_basis",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
    "skill_rank",
    "demand_category",
]


for column in required_columns:

    if column not in gap_df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


print(
    "\n[PASS] All required columns are present."
)


# ---------------------------------------------------------------
# 2. Dataset must not be empty
# ---------------------------------------------------------------

if gap_df.empty:

    raise ValueError(
        "Personal skill gap dataset is empty."
    )


print(
    "[PASS] Personal skill gap dataset is not empty."
)


# ---------------------------------------------------------------
# 3. Every record must be a user skill gap
# ---------------------------------------------------------------

if not (
    gap_df["user_skill_status"]
    == "Gap"
).all():

    raise ValueError(
        "Personal gap dataset contains a "
        "non-gap user skill."
    )


if not (
    gap_df["personal_gap"]
).all():

    raise ValueError(
        "Personal gap dataset contains a "
        "record where personal_gap is False."
    )


print(
    "[PASS] All records represent personal skill gaps."
)


# ---------------------------------------------------------------
# 4. Every gap must be market observed
# ---------------------------------------------------------------

comparison_market_check = comparison_df[
    [
        "occupation_uri",
        "skill_uri",
        "market_observed",
    ]
].drop_duplicates(
    subset=[
        "occupation_uri",
        "skill_uri",
    ]
)


gap_check = gap_df.merge(
    comparison_market_check,
    on=[
        "occupation_uri",
        "skill_uri",
    ],
    how="left",
)


if gap_check["market_observed"].isna().any():

    raise ValueError(
        "A personal gap could not be matched "
        "to the comparison dataset."
    )


if not (
    gap_check["market_observed"]
).all():

    raise ValueError(
        "Personal gap dataset contains "
        "a skill not observed in the market."
    )


print(
    "[PASS] All personal gaps are market-observed."
)


# ---------------------------------------------------------------
# 5. No duplicate occupation-skill pairs
# ---------------------------------------------------------------

if gap_df.duplicated(
    subset=[
        "occupation_uri",
        "skill_uri",
    ]
).any():

    raise ValueError(
        "Duplicate occupation-skill pairs found."
    )


print(
    "[PASS] No duplicate occupation-skill pairs."
)


# ---------------------------------------------------------------
# 6. Market prevalence must be valid
# ---------------------------------------------------------------

if not (
    gap_df["skill_prevalence_percent"]
    >= 0
).all():

    raise ValueError(
        "Negative market prevalence found."
    )


if not (
    gap_df["skill_prevalence_percent"]
    <= 100
).all():

    raise ValueError(
        "Market prevalence greater than 100% found."
    )


print(
    "[PASS] Market prevalence values are valid."
)


# ---------------------------------------------------------------
# 7. Market rank must exist for observed skills
# ---------------------------------------------------------------

if gap_df["skill_rank"].isna().any():

    raise ValueError(
        "Missing market rank found for "
        "a market-observed personal gap."
    )


print(
    "[PASS] All personal gaps have market ranks."
)


# ---------------------------------------------------------------
# 8. Evidence labels
# ---------------------------------------------------------------

expected_gap_type = (
    "Market-observed occupation skill not "
    "currently declared by user"
)

if not (
    gap_df["gap_type"]
    == expected_gap_type
).all():

    raise ValueError(
        "Unexpected gap_type value found."
    )


expected_basis = (
    "ESCO occupation representation + "
    "NYC market skill-demand evidence"
)

if not (
    gap_df["evidence_basis"]
    == expected_basis
).all():

    raise ValueError(
        "Unexpected evidence_basis value found."
    )


print(
    "[PASS] Gap evidence labels are consistent."
)


# ---------------------------------------------------------------
# 9. Check against original comparison
# ---------------------------------------------------------------

expected_pairs = set(
    comparison_df.loc[
        (
            comparison_df["user_skill_status"]
            == "Gap"
        )
        &
        (
            comparison_df["market_observed"]
            == True
        ),
        [
            "occupation_uri",
            "skill_uri",
        ],
    ].apply(
        tuple,
        axis=1,
    )
)

actual_pairs = set(
    gap_df[
        [
            "occupation_uri",
            "skill_uri",
        ]
    ].apply(
        tuple,
        axis=1,
    )
)


if actual_pairs != expected_pairs:

    missing = expected_pairs - actual_pairs
    extra = actual_pairs - expected_pairs

    raise ValueError(
        f"Personal gap records do not exactly match "
        f"the market-relevant comparison records. "
        f"Missing={len(missing)}, "
        f"Extra={len(extra)}"
    )


print(
    "[PASS] Personal gaps exactly match the "
    "market-relevant comparison subset."
)


# ---------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------

print("\nValidated personal gap summary:")

for occupation, group in gap_df.groupby(
    "occupation_label"
):

    print(
        f"  {occupation}: "
        f"{len(group)} market-relevant gaps"
    )


print(
    f"\nTotal validated personal gaps: "
    f"{len(gap_df)}"
)

print(
    "\nPHASE 6.5 VALIDATION: ALL CHECKS PASSED"
)