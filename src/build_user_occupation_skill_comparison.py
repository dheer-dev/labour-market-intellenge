import pandas as pd
from pathlib import Path


USER_SKILLS_FILE = Path(
    "data/processed/user_skill_representation.csv"
)

ESCO_RELATIONS_FILE = Path(
    "data/processed/esco_occupation_skill_relations.csv"
)

MARKET_SKILLS_FILE = Path(
    "data/processed/occupation_skill_demand_analysis.csv"
)

OUTPUT_FILE = Path(
    "data/processed/user_occupation_skill_comparison.csv"
)


TARGET_OCCUPATIONS = [
    "civil engineer",
    "construction manager",
    "environmental engineer",
    "lawyer",
    "project manager",
]


print("=" * 70)
print("PHASE 6.4 — USER–OCCUPATION SKILL COMPARISON")
print("=" * 70)


# ------------------------------------------------------------------
# 1. Load user skill representation
# ------------------------------------------------------------------

user_df = pd.read_csv(USER_SKILLS_FILE)

print(
    f"\nUser skill representation shape: "
    f"{user_df.shape}"
)


# ------------------------------------------------------------------
# 2. Load ESCO occupation-skill relations
# ------------------------------------------------------------------

esco_df = pd.read_csv(ESCO_RELATIONS_FILE)

print(
    f"ESCO occupation-skill relations shape: "
    f"{esco_df.shape}"
)


# ------------------------------------------------------------------
# 3. Load market skill-demand evidence
# ------------------------------------------------------------------

market_df = pd.read_csv(MARKET_SKILLS_FILE)

print(
    f"Market skill-demand shape: "
    f"{market_df.shape}"
)


# ------------------------------------------------------------------
# 4. Validate required columns
# ------------------------------------------------------------------

user_required = [
    "user_id",
    "user_skill",
    "esco_skill_uri",
    "esco_skill_label",
    "skill_possession",
]

esco_required = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
]

market_required = [
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
    "skill_rank",
    "demand_category",
]


for column in user_required:
    if column not in user_df.columns:
        raise ValueError(
            f"Missing user column: {column}"
        )


for column in esco_required:
    if column not in esco_df.columns:
        raise ValueError(
            f"Missing ESCO column: {column}"
        )


for column in market_required:
    if column not in market_df.columns:
        raise ValueError(
            f"Missing market column: {column}"
        )


print("\n[PASS] Required columns are present.")


# ------------------------------------------------------------------
# 5. Restrict ESCO relations to the five selected occupations
# ------------------------------------------------------------------

esco_target = esco_df[
    esco_df["occupation_label"]
    .str.lower()
    .isin(TARGET_OCCUPATIONS)
].copy()


if esco_target.empty:
    raise ValueError(
        "No ESCO occupation relations found "
        "for the selected occupations."
    )


print(
    f"\nESCO target occupation-skill rows: "
    f"{len(esco_target)}"
)


# ------------------------------------------------------------------
# 6. Remove duplicate occupation-skill relations
# ------------------------------------------------------------------

esco_target = esco_target.drop_duplicates(
    subset=[
        "occupation_uri",
        "skill_uri",
    ]
).copy()


print(
    f"Unique occupation-skill relations: "
    f"{len(esco_target)}"
)


# ------------------------------------------------------------------
# 7. Prepare user skill set
# ------------------------------------------------------------------

user_skill_uris = set(
    user_df.loc[
        user_df["skill_possession"] == 1,
        "esco_skill_uri",
    ]
)


if not user_skill_uris:
    raise ValueError(
        "No possessed user skills found."
    )


print(
    f"User possessed ESCO skills: "
    f"{len(user_skill_uris)}"
)


# ------------------------------------------------------------------
# 8. Build occupation-level comparison
# ------------------------------------------------------------------

comparison_rows = []


for occupation_uri, occupation_group in esco_target.groupby(
    "occupation_uri"
):

    occupation_label = (
        occupation_group["occupation_label"]
        .iloc[0]
    )

    for _, row in occupation_group.iterrows():

        skill_uri = row["skill_uri"]

        if skill_uri in user_skill_uris:
            skill_status = "Possessed"
        else:
            skill_status = "Gap"

        comparison_rows.append(
            {
                "user_id": user_df["user_id"].iloc[0],
                "occupation_uri": occupation_uri,
                "occupation_label": occupation_label,
                "skill_uri": skill_uri,
                "skill_label": row["skill_label"],
                "user_skill_status": skill_status,
            }
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


if comparison_df.empty:
    raise ValueError(
        "User–occupation comparison is empty."
    )


print(
    f"\nInitial comparison rows: "
    f"{len(comparison_df)}"
)


# ------------------------------------------------------------------
# 9. Add market evidence
# ------------------------------------------------------------------

market_subset = market_df[
    market_df["occupation_label"]
    .str.lower()
    .isin(TARGET_OCCUPATIONS)
].copy()


market_subset = market_subset[
    [
        "occupation_label",
        "skill_uri",
        "skill_label",
        "skill_posting_count",
        "postings_with_skills",
        "skill_prevalence",
        "skill_prevalence_percent",
        "skill_rank",
        "demand_category",
    ]
].drop_duplicates(
    subset=[
        "occupation_label",
        "skill_uri",
    ]
)


comparison_df = comparison_df.merge(
    market_subset,
    on=[
        "occupation_label",
        "skill_uri",
    ],
    how="left",
    suffixes=(
        "",
        "_market",
    ),
)


# ------------------------------------------------------------------
# 10. Add market evidence status
# ------------------------------------------------------------------

comparison_df["market_observed"] = (
    comparison_df["skill_posting_count"]
    .notna()
)


comparison_df["market_evidence"] = (
    comparison_df["market_observed"]
    .map(
        {
            True: "Observed in market postings",
            False: "Not observed in current market dataset",
        }
    )
)


# ------------------------------------------------------------------
# 11. Classify personal gap evidence
# ------------------------------------------------------------------

comparison_df["personal_gap"] = (
    comparison_df["user_skill_status"]
    == "Gap"
)


comparison_df["gap_evidence"] = comparison_df.apply(
    lambda row:
        (
            "Occupation skill not currently declared "
            "by user"
            if row["personal_gap"]
            else
            "Occupation skill currently declared "
            "by user"
        ),
    axis=1,
)


# ------------------------------------------------------------------
# 12. Order columns
# ------------------------------------------------------------------

comparison_df = comparison_df[
    [
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
]


# ------------------------------------------------------------------
# 13. Save
# ------------------------------------------------------------------

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ------------------------------------------------------------------
# 14. Summary
# ------------------------------------------------------------------

print("\nComparison summary:")

for occupation in TARGET_OCCUPATIONS:

    occupation_rows = comparison_df[
        comparison_df["occupation_label"]
        .str.lower()
        == occupation
    ]

    if occupation_rows.empty:
        print(
            f"  {occupation}: NO DATA"
        )
        continue

    possessed = (
        occupation_rows["user_skill_status"]
        == "Possessed"
    ).sum()

    gaps = (
        occupation_rows["user_skill_status"]
        == "Gap"
    ).sum()

    market_observed = (
        occupation_rows["market_observed"]
    ).sum()

    print(
        f"  {occupation}: "
        f"{len(occupation_rows)} ESCO skills | "
        f"{possessed} possessed | "
        f"{gaps} gaps | "
        f"{market_observed} market-observed"
    )


print(
    f"\nOutput: {OUTPUT_FILE}"
)

print(
    "\nPHASE 6.4: USER–OCCUPATION "
    "SKILL COMPARISON BUILD COMPLETE"
)