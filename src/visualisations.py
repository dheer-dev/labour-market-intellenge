import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

ESCO_RELATIONS = "data/processed/esco_occupation_skill_relations.csv"
MARKET_DEMAND = "data/processed/occupation_skill_demand_matrix.csv"
PERSONAL_GAPS = "data/processed/personal_skill_gaps.csv"
MODEL_SUMMARY = "data/processed/model_summary.csv"

SELECTED_OCCUPATIONS = [
    "civil engineer",
    "construction manager",
    "environmental engineer",
    "lawyer",
    "project manager"
]

TOP_N = 5


# ============================================================
# LOAD DATA
# ============================================================

esco = pd.read_csv(ESCO_RELATIONS)
market = pd.read_csv(MARKET_DEMAND)
personal_gaps = pd.read_csv(PERSONAL_GAPS)
model_summary = pd.read_csv(MODEL_SUMMARY)


# ============================================================
# 1. DATASET / COVERAGE OVERVIEW
# ============================================================

esco_pairs = esco[
    ["occupation_uri", "skill_uri"]
].drop_duplicates()

market_pairs = market[
    ["occupation_uri", "skill_uri"]
].drop_duplicates()

overlap_pairs = esco_pairs.merge(
    market_pairs,
    on=["occupation_uri", "skill_uri"],
    how="inner"
)

coverage_labels = [
    "ESCO occupation-skill pairs",
    "Market-observed pairs",
    "ESCO-market overlap",
    "Personal market-relevant gaps"
]

coverage_values = [
    len(esco_pairs),
    len(market_pairs),
    len(overlap_pairs),
    len(personal_gaps)
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    coverage_labels,
    coverage_values
)

plt.ylabel("Number of occupation-skill pairs")
plt.title("Labour Market Dataset Coverage")

plt.xticks(
    rotation=15,
    ha="right"
)

for bar, value in zip(bars, coverage_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "dataset_coverage_overview.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 2. TOP SKILLS BY OCCUPATION
# ============================================================

top_skills = (
    market[
        market["occupation_label"].isin(SELECTED_OCCUPATIONS)
    ]
    .sort_values(
        ["occupation_label", "skill_prevalence"],
        ascending=[True, False]
    )
    .groupby(
        "occupation_label",
        group_keys=False
    )
    .head(TOP_N)
    .copy()
)

top_skills["skill_display"] = (
    top_skills["skill_label"]
    .str.replace("_", " ", regex=False)
    .str.title()
)

fig, axes = plt.subplots(
    nrows=len(SELECTED_OCCUPATIONS),
    ncols=1,
    figsize=(12, 18)
)

for ax, occupation in zip(
    axes,
    SELECTED_OCCUPATIONS
):

    occupation_data = top_skills[
        top_skills["occupation_label"] == occupation
    ].sort_values("skill_prevalence")

    ax.barh(
        occupation_data["skill_display"],
        occupation_data["skill_prevalence_percent"]
    )

    ax.set_title(
        occupation.title(),
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Skill prevalence (%)")
    ax.set_ylabel("")

    for index, value in enumerate(
        occupation_data["skill_prevalence_percent"]
    ):
        ax.text(
            value + 0.5,
            index,
            f"{value:.1f}%",
            va="center"
        )

fig.suptitle(
    "Top 5 Market-Demanded Skills by Occupation",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

plt.savefig(
    FIGURES_DIR / "top_skills_by_occupation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 3. SKILL PREVALENCE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

plt.hist(
    market["skill_prevalence_percent"],
    bins=30
)

plt.xlabel("Skill prevalence (%)")
plt.ylabel("Number of occupation-skill pairs")
plt.title(
    "Distribution of Skill Prevalence Across Market-Observed Pairs"
)

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "skill_prevalence_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. ESCO VS MARKET COMPARISON
# ============================================================

esco_counts = (
    esco.groupby("occupation_label")["skill_uri"]
    .nunique()
    .rename("esco_skill_count")
)

market_counts = (
    market.groupby("occupation_label")["skill_uri"]
    .nunique()
    .rename("market_skill_count")
)

comparison = pd.concat(
    [esco_counts, market_counts],
    axis=1
).fillna(0)

comparison = comparison.loc[
    comparison.index.intersection(SELECTED_OCCUPATIONS)
]

comparison = comparison.reindex(
    SELECTED_OCCUPATIONS
)

x = range(len(comparison))
width = 0.35

plt.figure(figsize=(12, 7))

plt.bar(
    [i - width / 2 for i in x],
    comparison["esco_skill_count"],
    width,
    label="ESCO"
)

plt.bar(
    [i + width / 2 for i in x],
    comparison["market_skill_count"],
    width,
    label="NYC Market"
)

plt.xticks(
    list(x),
    [
        occupation.title()
        for occupation in comparison.index
    ],
    rotation=15,
    ha="right"
)

plt.ylabel("Number of skills")
plt.title("ESCO vs Market-Observed Skill Coverage")
plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "esco_vs_market_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. PERSONAL SKILL-GAP RANKING
# ============================================================

personal_plot = personal_gaps.copy()

personal_plot["skill_display"] = (
    personal_plot["skill_label"]
    .str.replace("_", " ", regex=False)
    .str.title()
)

fig, axes = plt.subplots(
    nrows=len(SELECTED_OCCUPATIONS),
    ncols=1,
    figsize=(12, 22)
)

for ax, occupation in zip(
    axes,
    SELECTED_OCCUPATIONS
):

    occupation_data = (
        personal_plot[
            personal_plot["occupation_label"] == occupation
        ]
        .sort_values(
            "skill_prevalence_percent",
            ascending=True
        )
        .tail(10)
    )

    ax.barh(
        occupation_data["skill_display"],
        occupation_data["skill_prevalence_percent"]
    )

    ax.set_title(
        occupation.title(),
        fontsize=12,
        fontweight="bold"
    )

    ax.set_xlabel("Market prevalence (%)")
    ax.set_ylabel("")

    for index, value in enumerate(
        occupation_data["skill_prevalence_percent"]
    ):
        ax.text(
            value + 0.3,
            index,
            f"{value:.1f}%",
            va="center"
        )

fig.suptitle(
    "Top Market-Relevant Personal Skill Gaps",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout(
    rect=[0, 0, 1, 0.97]
)

plt.savefig(
    FIGURES_DIR / "personal_skill_gap_ranking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. MODEL EVALUATION
# ============================================================

# model_summary.csv is stored in wide format.
# Each metric is a separate column.

required_model_columns = [
    "cv_accuracy_mean",
    "cv_precision_mean",
    "cv_recall_mean",
    "cv_f1_mean",
    "cv_roc_auc_mean",
    "cv_pr_auc_mean"
]

missing_columns = [
    column
    for column in required_model_columns
    if column not in model_summary.columns
]

if missing_columns:
    raise ValueError(
        "Missing model summary columns: "
        + ", ".join(missing_columns)
    )

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "PR-AUC"
]

metric_columns = required_model_columns

cv_means = [
    float(model_summary[column].iloc[0])
    for column in metric_columns
]

plt.figure(figsize=(10, 6))

bars = plt.bar(
    metric_names,
    cv_means
)

plt.ylim(0, 1)

plt.ylabel("Score")
plt.title(
    "Model Evaluation: 5-Fold Cross-Validation"
)

for bar, value in zip(
    bars,
    cv_means
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value,
        f"{value:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "model_evaluation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("REVIEW 2 VISUALISATIONS CREATED")
print("=" * 70)

print("\n1. Dataset coverage")
print(f"   ESCO pairs: {len(esco_pairs):,}")
print(f"   Market pairs: {len(market_pairs):,}")
print(f"   ESCO-market overlap: {len(overlap_pairs):,}")
print(f"   Personal market-relevant gaps: {len(personal_gaps):,}")

print("\n2. Top skills by occupation")

for occupation in SELECTED_OCCUPATIONS:

    print(f"\n   {occupation.title()}:")

    occupation_data = top_skills[
        top_skills["occupation_label"] == occupation
    ].sort_values(
        "skill_prevalence",
        ascending=False
    )

    for _, row in occupation_data.iterrows():

        print(
            f"      {row['skill_label']}: "
            f"{row['skill_prevalence_percent']:.2f}%"
        )

print("\n3. Skill prevalence distribution")
print("   Created successfully")

print("\n4. ESCO vs market comparison")
print("   Created successfully")

print("\n5. Personal skill-gap ranking")
print(
    f"   Total personal market-relevant gaps: "
    f"{len(personal_gaps)}"
)

print("\n6. Model evaluation")
print("   Created successfully")

print("\nSaved figures:")

for filename in [
    "dataset_coverage_overview.png",
    "top_skills_by_occupation.png",
    "skill_prevalence_distribution.png",
    "esco_vs_market_comparison.png",
    "personal_skill_gap_ranking.png",
    "model_evaluation.png"
]:
    print(f"   figures/{filename}")

print("=" * 70)