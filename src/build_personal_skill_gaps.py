import pandas as pd
from pathlib import Path


COMPARISON_FILE = Path(
    "data/processed/user_occupation_skill_comparison.csv"
)

OUTPUT_FILE = Path(
    "data/processed/personal_skill_gaps.csv"
)


comparison_df = pd.read_csv(COMPARISON_FILE)


print("=" * 70)
print("PHASE 6.5 — BUILD MARKET-RELEVANT PERSONAL SKILL GAPS")
print("=" * 70)


print(
    f"\nUser–occupation comparison shape: "
    f"{comparison_df.shape}"
)


# ------------------------------------------------------------------
# 1. Validate required columns
# ------------------------------------------------------------------

required_columns = [
    "user_id",
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "user_skill_status",
    "personal_gap",
    "market_observed",
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


print(
    "\n[PASS] Required columns are present."
)


# ------------------------------------------------------------------
# 2. Select only skills that:
#
#    a) are not currently possessed by the user
#    b) are observed in market postings
# ------------------------------------------------------------------

gap_df = comparison_df[
    (
        comparison_df["user_skill_status"]
        == "Gap"
    )
    &
    (
        comparison_df["market_observed"]
        == True
    )
].copy()


if gap_df.empty:

    raise ValueError(
        "No market-relevant personal skill gaps found."
    )


print(
    f"\nMarket-relevant personal gaps: "
    f"{len(gap_df)}"
)


# ------------------------------------------------------------------
# 3. Add explicit evidence classification
# ------------------------------------------------------------------

gap_df["gap_type"] = (
    "Market-observed occupation skill not "
    "currently declared by user"
)


gap_df["evidence_basis"] = (
    "ESCO occupation representation + "
    "NYC market skill-demand evidence"
)


# ------------------------------------------------------------------
# 4. Preserve the existing market evidence
# ------------------------------------------------------------------

gap_df = gap_df[
    [
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
].copy()


# ------------------------------------------------------------------
# 5. Sort by occupation and market prevalence
# ------------------------------------------------------------------

gap_df = gap_df.sort_values(
    by=[
        "occupation_label",
        "skill_prevalence_percent",
        "skill_rank",
    ],
    ascending=[
        True,
        False,
        True,
    ],
).reset_index(drop=True)


# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------

gap_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------------
# 7. Occupation-level summary
# ------------------------------------------------------------------

print("\nPersonal gap summary:")

for occupation, group in gap_df.groupby(
    "occupation_label"
):

    print(
        f"  {occupation}: "
        f"{len(group)} market-relevant gaps"
    )


# ------------------------------------------------------------------
# 8. Show top market-observed gaps
# ------------------------------------------------------------------

print(
    "\nTop market-observed personal gaps:"
)

for occupation, group in gap_df.groupby(
    "occupation_label"
):

    print(
        f"\n{occupation}:"
    )

    for _, row in group.head(5).iterrows():

        print(
            f"  {row['skill_label']} | "
            f"{row['skill_prevalence_percent']:.2f}% "
            f"prevalence | "
            f"rank #{int(row['skill_rank'])}"
        )


print(
    f"\nOutput: {OUTPUT_FILE}"
)

print(
    "\nPHASE 6.5: MARKET-RELEVANT PERSONAL "
    "SKILL GAP BUILD COMPLETE"
)