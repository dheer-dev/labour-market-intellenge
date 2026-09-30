from pathlib import Path

import pandas as pd


# ============================================================
# PHASE 5 — SKILL-GAP ENGINE
# Step 3: Prioritisation Rule Analysis
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "occupation_skill_gap_analysis.csv"
)


# ------------------------------------------------------------
# 1. Load Phase 4 output
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PHASE 5 — PRIORITISATION RULE ANALYSIS")
print("STEP 3 — EMPIRICAL RULE ANALYSIS")
print("=" * 70)

print(f"\nInput shape: {df.shape}")


# ------------------------------------------------------------
# 2. Select the same candidate population as Step 2
# ------------------------------------------------------------

candidate_gaps = df[
    (df["gap_category"] == "Market-only")
    & (df["market_supported"] == True)
].copy()

print(f"\nCandidate gap count: {len(candidate_gaps)}")


# ------------------------------------------------------------
# 3. Basic candidate-gap distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. CANDIDATE GAP DISTRIBUTION")
print("=" * 70)

print("\nCandidate gaps by occupation:")
print(
    candidate_gaps["occupation_label"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 4. Prevalence distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MARKET PREVALENCE DISTRIBUTION")
print("=" * 70)

prevalence = candidate_gaps["market_skill_prevalence_percent"]

print(
    prevalence.describe()[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
)


# ------------------------------------------------------------
# 5. Prevalence bands
# ------------------------------------------------------------

def prevalence_band(value):
    if value >= 40:
        return ">= 40%"
    elif value >= 30:
        return "30–<40%"
    elif value >= 20:
        return "20–<30%"
    else:
        return "<20%"


candidate_gaps["prevalence_band"] = (
    candidate_gaps["market_skill_prevalence_percent"]
    .apply(prevalence_band)
)

print("\nPrevalence bands:")
print(
    candidate_gaps["prevalence_band"]
    .value_counts()
    .reindex(
        [">= 40%", "30–<40%", "20–<30%", "<20%"],
        fill_value=0,
    )
)


# ------------------------------------------------------------
# 6. Rank distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. OCCUPATION-RELATIVE RANK DISTRIBUTION")
print("=" * 70)

rank = candidate_gaps["market_skill_rank"]

print(
    rank.describe()[
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
)


# ------------------------------------------------------------
# 7. Rank bands
# ------------------------------------------------------------

def rank_band(value):
    if value <= 3:
        return "Rank 1–3"
    elif value <= 6:
        return "Rank 4–6"
    else:
        return "Rank 7+"


candidate_gaps["rank_band"] = (
    candidate_gaps["market_skill_rank"]
    .apply(rank_band)
)

print("\nRank bands:")
print(
    candidate_gaps["rank_band"]
    .value_counts()
    .reindex(
        ["Rank 1–3", "Rank 4–6", "Rank 7+"],
        fill_value=0,
    )
)


# ------------------------------------------------------------
# 8. Demand-category distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. MARKET DEMAND CATEGORY DISTRIBUTION")
print("=" * 70)

print(
    candidate_gaps["market_demand_category"]
    .value_counts()
)


# ------------------------------------------------------------
# 9. Cross-tab: prevalence vs rank
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. PREVALENCE BAND × RANK BAND")
print("=" * 70)

prevalence_rank_table = pd.crosstab(
    candidate_gaps["prevalence_band"],
    candidate_gaps["rank_band"],
)

prevalence_rank_table = prevalence_rank_table.reindex(
    index=[">= 40%", "30–<40%", "20–<30%", "<20%"],
    columns=["Rank 1–3", "Rank 4–6", "Rank 7+"],
    fill_value=0,
)

print(prevalence_rank_table)


# ------------------------------------------------------------
# 10. Cross-tab: prevalence vs demand category
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. PREVALENCE BAND × DEMAND CATEGORY")
print("=" * 70)

prevalence_demand_table = pd.crosstab(
    candidate_gaps["prevalence_band"],
    candidate_gaps["market_demand_category"],
)

print(prevalence_demand_table)


# ------------------------------------------------------------
# 11. Cross-tab: rank vs demand category
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. RANK BAND × DEMAND CATEGORY")
print("=" * 70)

rank_demand_table = pd.crosstab(
    candidate_gaps["rank_band"],
    candidate_gaps["market_demand_category"],
)

print(rank_demand_table)


# ------------------------------------------------------------
# 12. Correlation between prevalence and rank
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10. PREVALENCE–RANK RELATIONSHIP")
print("=" * 70)

correlation = candidate_gaps[
    [
        "market_skill_prevalence_percent",
        "market_skill_rank",
    ]
].corr(method="spearman")

print("\nSpearman correlation:")
print(correlation)


# ------------------------------------------------------------
# 13. Occupation-level distributions
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("11. OCCUPATION-LEVEL PREVALENCE DISTRIBUTION")
print("=" * 70)

occupation_summary = (
    candidate_gaps
    .groupby("occupation_label")
    .agg(
        candidate_gap_count=("skill_label", "count"),
        min_prevalence=(
            "market_skill_prevalence_percent",
            "min",
        ),
        median_prevalence=(
            "market_skill_prevalence_percent",
            "median",
        ),
        max_prevalence=(
            "market_skill_prevalence_percent",
            "max",
        ),
        min_rank=("market_skill_rank", "min"),
        median_rank=("market_skill_rank", "median"),
        max_rank=("market_skill_rank", "max"),
    )
    .sort_index()
)

print(occupation_summary.to_string())


# ------------------------------------------------------------
# 14. Borderline candidates around existing thresholds
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("12. THRESHOLD-BORDERLINE CANDIDATES")
print("=" * 70)

# Existing Step 2 thresholds:
# High      >= 40%
# Moderate  >= 30%
# Emerging  < 30%

borderline = candidate_gaps[
    (
        candidate_gaps["market_skill_prevalence_percent"].between(
            28, 32
        )
    )
    |
    (
        candidate_gaps["market_skill_prevalence_percent"].between(
            38, 42
        )
    )
].copy()

borderline_columns = [
    "occupation_label",
    "skill_label",
    "market_skill_posting_count",
    "market_skill_prevalence_percent",
    "market_skill_rank",
    "market_demand_category",
    "prevalence_band",
    "rank_band",
]

print(
    borderline[
        borderline_columns
    ]
    .sort_values(
        [
            "market_skill_prevalence_percent",
            "occupation_label",
        ],
        ascending=[False, True],
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# 15. Identify high-prevalence but lower-ranked cases
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("13. PREVALENCE–RANK DISAGREEMENT CASES")
print("=" * 70)

# High prevalence but rank outside top 3
high_prevalence_lower_rank = candidate_gaps[
    (
        candidate_gaps["market_skill_prevalence_percent"] >= 40
    )
    &
    (
        candidate_gaps["market_skill_rank"] > 3
    )
].copy()

print("\nHigh prevalence (>=40%) but rank > 3:")

if len(high_prevalence_lower_rank) == 0:
    print("None")
else:
    print(
        high_prevalence_lower_rank[
            [
                "occupation_label",
                "skill_label",
                "market_skill_posting_count",
                "market_skill_prevalence_percent",
                "market_skill_rank",
                "market_demand_category",
            ]
        ]
        .sort_values(
            "market_skill_prevalence_percent",
            ascending=False,
        )
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 16. Top-ranked but lower-prevalence cases
# ------------------------------------------------------------

top_rank_lower_prevalence = candidate_gaps[
    (
        candidate_gaps["market_skill_rank"] <= 3
    )
    &
    (
        candidate_gaps["market_skill_prevalence_percent"] < 30
    )
].copy()

print("\nTop-3 rank but prevalence <30%:")

if len(top_rank_lower_prevalence) == 0:
    print("None")
else:
    print(
        top_rank_lower_prevalence[
            [
                "occupation_label",
                "skill_label",
                "market_skill_posting_count",
                "market_skill_prevalence_percent",
                "market_skill_rank",
                "market_demand_category",
            ]
        ]
        .sort_values(
            [
                "occupation_label",
                "market_skill_rank",
            ]
        )
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 17. Full evidence table
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("14. FULL CANDIDATE EVIDENCE TABLE")
print("=" * 70)

evidence_columns = [
    "occupation_label",
    "skill_label",
    "market_skill_posting_count",
    "market_skill_prevalence_percent",
    "market_skill_rank",
    "market_demand_category",
    "prevalence_band",
    "rank_band",
]

print(
    candidate_gaps[
        evidence_columns
    ]
    .sort_values(
        [
            "occupation_label",
            "market_skill_prevalence_percent",
        ],
        ascending=[True, False],
    )
    .to_string(index=False)
)


# ------------------------------------------------------------
# 18. Completion
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PHASE 5 STEP 3 COMPLETE")
print("=" * 70)

print(
    "\nNo final prioritisation rule has been frozen yet."
    "\nThe results above will be used to justify the final rule."
)