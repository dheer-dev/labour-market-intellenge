import pandas as pd
from pathlib import Path


# ==============================================================
# PHASE 6.1 — BUILD EXPLAINABLE SKILL-GAP EVIDENCE DATASET
# ==============================================================

SOURCE_PATH = Path(
    "data/processed/occupation_skill_gap_analysis.csv"
)

PRIORITY_PATH = Path(
    "data/processed/skill_gap_prioritisation.csv"
)

OUTPUT_PATH = Path(
    "data/processed/skill_gap_explanations.csv"
)


print("=" * 70)
print("PHASE 6.1 — EXPLAINABLE SKILL-GAP EVIDENCE")
print("=" * 70)


# --------------------------------------------------------------
# 1. LOAD INPUTS
# --------------------------------------------------------------

source_df = pd.read_csv(SOURCE_PATH)
priority_df = pd.read_csv(PRIORITY_PATH)

print(f"\nSource shape:       {source_df.shape}")
print(f"Priority shape:     {priority_df.shape}")


# --------------------------------------------------------------
# 2. RECREATE THE VALIDATED CANDIDATE SET
# --------------------------------------------------------------

candidate_df = source_df[
    (source_df["gap_category"] == "Market-only")
    & (source_df["market_supported"] == True)
].copy()

print(f"Candidate gaps:     {len(candidate_df)}")


# --------------------------------------------------------------
# 3. SELECT SOURCE EVIDENCE
# --------------------------------------------------------------

evidence_columns = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "market_skill_posting_count",
    "market_skill_prevalence",
    "market_skill_prevalence_percent",
    "market_skill_rank",
    "market_demand_category",
    "esco_relation_type",
    "esco_skill_type",
    "gap_category",
    "market_supported",
]

evidence_df = candidate_df[evidence_columns].copy()


# --------------------------------------------------------------
# 4. CHECK UNIQUENESS BEFORE MERGING
# --------------------------------------------------------------

candidate_duplicates = evidence_df.duplicated(
    subset=["occupation_label", "skill_label"]
).sum()

priority_duplicates = priority_df.duplicated(
    subset=["occupation_label", "skill_label"]
).sum()

print("\nDuplicate checks:")
print(f"Source candidate duplicates:   {candidate_duplicates}")
print(f"Priority output duplicates:    {priority_duplicates}")

if candidate_duplicates != 0:
    raise ValueError(
        "Duplicate occupation-skill pairs found in candidate evidence."
    )

if priority_duplicates != 0:
    raise ValueError(
        "Duplicate occupation-skill pairs found in prioritisation output."
    )


# --------------------------------------------------------------
# 5. MERGE PRIORITISATION WITH SOURCE EVIDENCE
# --------------------------------------------------------------

priority_columns = [
    "occupation_label",
    "skill_label",
    "rank_band",
    "priority",
    "priority_explanation",
]

priority_subset = priority_df[priority_columns].copy()

explanation_df = evidence_df.merge(
    priority_subset,
    on=["occupation_label", "skill_label"],
    how="left",
    validate="one_to_one",
)


# --------------------------------------------------------------
# 6. VALIDATE THE MERGE
# --------------------------------------------------------------

missing_priority = explanation_df["priority"].isna().sum()

unexpected_rows = len(explanation_df) != len(priority_df)

print("\nMerge validation:")
print(f"Rows after merge:              {len(explanation_df)}")
print(f"Rows missing priority:         {missing_priority}")
print(f"Row-count mismatch:            {unexpected_rows}")

if missing_priority != 0:
    raise ValueError(
        "Some candidate gaps could not be matched to Phase 5 priorities."
    )

if unexpected_rows:
    raise ValueError(
        "Merged explanation dataset does not contain exactly 35 rows."
    )


# --------------------------------------------------------------
# 7. CREATE STRUCTURED GAP EXPLANATION
# --------------------------------------------------------------

def build_gap_explanation(row):
    prevalence = row["market_skill_prevalence_percent"]
    rank = int(row["market_skill_rank"])

    priority = row["priority"]

    if priority == "High":
        threshold_text = (
            "at or above the 40% high-priority threshold"
        )
    elif priority == "Moderate":
        threshold_text = (
            "within the 30–<40% moderate-priority prevalence band"
        )
    else:
        threshold_text = (
            "within the 20–<30% emerging-priority prevalence band"
        )

    return (
        f"{row['skill_label']} is a market-only skill gap for "
        f"{row['occupation_label']}. It appears in "
        f"{prevalence:.2f}% of relevant market postings "
        f"({int(row['market_skill_posting_count'])} postings). "
        f"It ranks {rank} among the observed market-supported "
        f"skill gaps for this occupation. "
        f"The skill is not represented in the corresponding ESCO "
        f"occupation-skill relations. "
        f"Under the validated prevalence-based prioritisation rule, "
        f"it is classified as {priority} because its prevalence is "
        f"{threshold_text}."
    )


explanation_df["evidence_explanation"] = explanation_df.apply(
    build_gap_explanation,
    axis=1
)


# --------------------------------------------------------------
# 8. ORDER COLUMNS
# --------------------------------------------------------------

output_columns = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "market_skill_posting_count",
    "market_skill_prevalence",
    "market_skill_prevalence_percent",
    "market_skill_rank",
    "market_demand_category",
    "esco_relation_type",
    "esco_skill_type",
    "gap_category",
    "market_supported",
    "rank_band",
    "priority",
    "priority_explanation",
    "evidence_explanation",
]

explanation_df = explanation_df[output_columns]


# --------------------------------------------------------------
# 9. SAVE
# --------------------------------------------------------------

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

explanation_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# --------------------------------------------------------------
# 10. SUMMARY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 6.1 SUMMARY")
print("=" * 70)

print(f"\nOutput shape: {explanation_df.shape}")

print("\nPriority distribution:")
print(
    explanation_df["priority"]
    .value_counts()
    .reindex(["High", "Moderate", "Emerging"])
)

print("\nGap category distribution:")
print(explanation_df["gap_category"].value_counts())

print("\nESCO relation values:")
print(explanation_df["esco_relation_type"].value_counts(dropna=False))

print("\nMarket-supported values:")
print(explanation_df["market_supported"].value_counts())

print(f"\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("PHASE 6.1 COMPLETE")
print("=" * 70)