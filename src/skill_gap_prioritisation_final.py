from pathlib import Path

import pandas as pd


# ============================================================
# PHASE 5 — SKILL-GAP ENGINE
# Step 4: Final Explainable Prioritisation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "occupation_skill_gap_analysis.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "skill_gap_prioritisation.csv"
)


# ------------------------------------------------------------
# 1. Load Phase 4 output
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PHASE 5 — FINAL SKILL-GAP PRIORITISATION")
print("=" * 70)

print(f"\nInput shape: {df.shape}")


# ------------------------------------------------------------
# 2. Select supported market-only candidate gaps
# ------------------------------------------------------------

candidate_gaps = df[
    (df["gap_category"] == "Market-only")
    & (df["market_supported"] == True)
].copy()

print(f"\nCandidate gap count: {len(candidate_gaps)}")


# ------------------------------------------------------------
# 3. Final prioritisation rule
# ------------------------------------------------------------

def final_priority(prevalence):
    if prevalence >= 40:
        return "High"
    elif prevalence >= 30:
        return "Moderate"
    else:
        return "Emerging"


candidate_gaps["priority"] = (
    candidate_gaps[
        "market_skill_prevalence_percent"
    ].apply(final_priority)
)


# ------------------------------------------------------------
# 4. Supporting evidence classification
# ------------------------------------------------------------

def rank_band(rank):
    if rank <= 3:
        return "Top 3"
    elif rank <= 6:
        return "Rank 4–6"
    else:
        return "Rank 7+"


candidate_gaps["rank_band"] = (
    candidate_gaps["market_skill_rank"]
    .apply(rank_band)
)


# ------------------------------------------------------------
# 5. Generate explanation
# ------------------------------------------------------------

def generate_explanation(row):

    prevalence = row["market_skill_prevalence_percent"]
    rank = row["market_skill_rank"]
    demand_category = row["market_demand_category"]
    priority = row["priority"]

    if priority == "High":
        explanation = (
            f"High priority because the skill appears in "
            f"{prevalence:.2f}% of market postings for this "
            f"occupation, exceeding the 40% high-priority "
            f"threshold."
        )

    elif priority == "Moderate":
        explanation = (
            f"Moderate priority because the skill appears in "
            f"{prevalence:.2f}% of market postings for this "
            f"occupation, placing it in the 30–<40% prevalence "
            f"band."
        )

    else:
        explanation = (
            f"Emerging priority because the skill appears in "
            f"{prevalence:.2f}% of market postings for this "
            f"occupation, placing it in the 20–<30% prevalence "
            f"band."
        )

    explanation += (
        f" It ranks {int(rank)} among observed market skills "
        f"for this occupation and is classified as "
        f"{demand_category} in the existing demand categories."
    )

    return explanation


candidate_gaps["priority_explanation"] = (
    candidate_gaps.apply(
        generate_explanation,
        axis=1
    )
)


# ------------------------------------------------------------
# 6. Select final output columns
# ------------------------------------------------------------

output_columns = [
    "occupation_label",
    "skill_label",
    "market_skill_posting_count",
    "market_skill_prevalence_percent",
    "market_skill_rank",
    "market_demand_category",
    "rank_band",
    "priority",
    "priority_explanation",
]

prioritised = candidate_gaps[
    output_columns
].copy()


# ------------------------------------------------------------
# 7. Sort by occupation and priority
# ------------------------------------------------------------

priority_order = {
    "High": 1,
    "Moderate": 2,
    "Emerging": 3,
}

prioritised["_priority_order"] = (
    prioritised["priority"]
    .map(priority_order)
)

prioritised = (
    prioritised
    .sort_values(
        [
            "occupation_label",
            "_priority_order",
            "market_skill_prevalence_percent",
        ],
        ascending=[True, True, False],
    )
    .drop(columns="_priority_order")
)


# ------------------------------------------------------------
# 8. Priority distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL PRIORITY DISTRIBUTION")
print("=" * 70)

print(
    prioritised["priority"]
    .value_counts()
    .reindex(
        ["High", "Moderate", "Emerging"],
        fill_value=0,
    )
)


# ------------------------------------------------------------
# 9. Priority distribution by occupation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PRIORITY DISTRIBUTION BY OCCUPATION")
print("=" * 70)

occupation_priority = pd.crosstab(
    prioritised["occupation_label"],
    prioritised["priority"],
)

occupation_priority = occupation_priority.reindex(
    columns=["High", "Moderate", "Emerging"],
    fill_value=0,
)

print(occupation_priority)


# ------------------------------------------------------------
# 10. Final prioritised gaps
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL PRIORITISED SKILL GAPS")
print("=" * 70)

print(
    prioritised.to_string(index=False)
)


# ------------------------------------------------------------
# 11. Save output
# ------------------------------------------------------------

prioritised.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 70)
print("OUTPUT SAVED")
print("=" * 70)

print(f"\n{OUTPUT_FILE}")

print("\n" + "=" * 70)
print("PHASE 5 STEP 4 COMPLETE")
print("=" * 70)