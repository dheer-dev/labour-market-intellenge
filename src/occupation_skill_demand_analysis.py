from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MATRIX_FILE = PROCESSED_DIR / "occupation_skill_demand_matrix.csv"
TARGET_FILE = PROCESSED_DIR / "target_occupation_esco_analysis.csv"

OUTPUT_FILE = PROCESSED_DIR / "occupation_skill_demand_analysis.csv"
SUMMARY_FILE = PROCESSED_DIR / "occupation_skill_demand_summary.csv"


# Selected occupations for the detailed analysis
SELECTED_OCCUPATIONS = [
    "project manager",
    "civil engineer",
    "lawyer",
    "environmental engineer",
    "construction manager",
]


# Project-defined descriptive thresholds
CORE_THRESHOLD = 0.50
COMMON_THRESHOLD = 0.20
OCCASIONAL_THRESHOLD = 0.05


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 3 — OCCUPATION-SPECIFIC SKILL DEMAND")
print("=" * 70)

matrix = pd.read_csv(MATRIX_FILE)
target = pd.read_csv(TARGET_FILE)

print(f"\nOccupation-skill matrix shape: {matrix.shape}")
print(f"Target occupation analysis shape: {target.shape}")


# ============================================================
# VALIDATE INPUT COLUMNS
# ============================================================

required_matrix_cols = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
]

missing = [
    col for col in required_matrix_cols
    if col not in matrix.columns
]

if missing:
    raise ValueError(
        f"Missing required columns in occupation_skill_demand_matrix.csv: {missing}"
    )


required_target_cols = [
    "occupation_uri",
    "occupation_label",
]

missing_target = [
    col for col in required_target_cols
    if col not in target.columns
]

if missing_target:
    raise ValueError(
        f"Missing required columns in target_occupation_esco_analysis.csv: "
        f"{missing_target}"
    )


# ============================================================
# NORMALIZE OCCUPATION LABELS
# ============================================================

matrix["occupation_label_norm"] = (
    matrix["occupation_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

target["occupation_label_norm"] = (
    target["occupation_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


selected_set = set(SELECTED_OCCUPATIONS)

missing_occupations = selected_set - set(
    target["occupation_label_norm"]
)

if missing_occupations:
    raise ValueError(
        f"Selected occupations not found in target analysis: "
        f"{sorted(missing_occupations)}"
    )


# ============================================================
# FILTER SELECTED OCCUPATIONS
# ============================================================

selected_matrix = matrix[
    matrix["occupation_label_norm"].isin(selected_set)
].copy()

selected_target = target[
    target["occupation_label_norm"].isin(selected_set)
].copy()


print(
    f"\nSelected occupations: "
    f"{selected_target['occupation_label'].nunique()}"
)

print(
    selected_target[
        ["occupation_label", "occupation_uri"]
    ].drop_duplicates().to_string(index=False)
)


# ============================================================
# VALIDATE UNIQUE OCCUPATION-SKILL PAIRS
# ============================================================

duplicate_pairs = selected_matrix.duplicated(
    subset=["occupation_uri", "skill_uri"]
).sum()

print(f"\nDuplicate occupation-skill pairs: {duplicate_pairs}")

if duplicate_pairs > 0:
    raise ValueError(
        "Duplicate occupation-skill pairs found."
    )


# ============================================================
# VALIDATE PREVALENCE
# ============================================================

selected_matrix["skill_prevalence"] = pd.to_numeric(
    selected_matrix["skill_prevalence"],
    errors="coerce"
)

selected_matrix["skill_prevalence_percent"] = pd.to_numeric(
    selected_matrix["skill_prevalence_percent"],
    errors="coerce"
)

selected_matrix["skill_posting_count"] = pd.to_numeric(
    selected_matrix["skill_posting_count"],
    errors="coerce"
)

selected_matrix["postings_with_skills"] = pd.to_numeric(
    selected_matrix["postings_with_skills"],
    errors="coerce"
)

if selected_matrix["skill_prevalence"].isna().any():
    raise ValueError(
        "Missing or invalid skill_prevalence values found."
    )

if not selected_matrix["skill_prevalence"].between(0, 1).all():
    raise ValueError(
        "skill_prevalence must be between 0 and 1."
    )


# ============================================================
# SKILL RANK
# ============================================================

selected_matrix = selected_matrix.sort_values(
    by=[
        "occupation_label",
        "skill_prevalence",
        "skill_posting_count",
        "skill_label",
    ],
    ascending=[
        True,
        False,
        False,
        True,
    ],
).reset_index(drop=True)


selected_matrix["skill_rank"] = (
    selected_matrix
    .groupby("occupation_uri")
    .cumcount()
    + 1
)


# ============================================================
# DEMAND CATEGORY
# ============================================================

def classify_demand(prevalence):
    if prevalence >= CORE_THRESHOLD:
        return "Core"
    elif prevalence >= COMMON_THRESHOLD:
        return "Common"
    elif prevalence >= OCCASIONAL_THRESHOLD:
        return "Occasional"
    else:
        return "Rare"


selected_matrix["demand_category"] = (
    selected_matrix["skill_prevalence"]
    .apply(classify_demand)
)


# ============================================================
# ADD USEFUL FLAGS
# ============================================================

selected_matrix["is_common_skill"] = (
    selected_matrix["skill_prevalence"]
    >= COMMON_THRESHOLD
)

selected_matrix["is_singleton_skill"] = (
    selected_matrix["skill_posting_count"]
    == 1
)


# ============================================================
# FINAL DETAILED OUTPUT
# ============================================================

output_columns = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
    "skill_rank",
    "demand_category",
    "is_common_skill",
    "is_singleton_skill",
]

detailed_output = selected_matrix[
    output_columns
].copy()


detailed_output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# OCCUPATION SUMMARY
# ============================================================

summary_rows = []

for occupation_uri, group in selected_matrix.groupby(
    "occupation_uri",
    sort=False
):

    occupation_label = group["occupation_label"].iloc[0]

    top_skills = (
        group.sort_values("skill_rank")
        .head(10)["skill_label"]
        .tolist()
    )

    summary_rows.append({
        "occupation_uri": occupation_uri,
        "occupation_label": occupation_label,

        "postings_with_skills":
            group["postings_with_skills"].iloc[0],

        "distinct_observed_skills":
            len(group),

        "core_skill_count":
            (group["demand_category"] == "Core").sum(),

        "common_skill_count":
            (group["demand_category"] == "Common").sum(),

        "occasional_skill_count":
            (group["demand_category"] == "Occasional").sum(),

        "rare_skill_count":
            (group["demand_category"] == "Rare").sum(),

        "singleton_skill_count":
            group["is_singleton_skill"].sum(),

        "top_1_skill":
            top_skills[0] if len(top_skills) > 0 else None,

        "top_1_skill_prevalence_percent":
            group.iloc[0]["skill_prevalence_percent"]
            if len(group) > 0 else None,

        "top_2_skill":
            top_skills[1] if len(top_skills) > 1 else None,

        "top_3_skill":
            top_skills[2] if len(top_skills) > 2 else None,

        "top_10_skills":
            " | ".join(top_skills),
    })


summary = pd.DataFrame(summary_rows)


# Keep the occupation order defined above
occupation_order = {
    occupation: i
    for i, occupation in enumerate(SELECTED_OCCUPATIONS)
}

summary["sort_order"] = (
    summary["occupation_label"]
    .str.lower()
    .map(occupation_order)
)

summary = (
    summary
    .sort_values("sort_order")
    .drop(columns=["sort_order"])
    .reset_index(drop=True)
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OCCUPATION SUMMARY")
print("=" * 70)

print(
    summary[
        [
            "occupation_label",
            "postings_with_skills",
            "distinct_observed_skills",
            "core_skill_count",
            "common_skill_count",
            "occasional_skill_count",
            "rare_skill_count",
            "singleton_skill_count",
        ]
    ].to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP 10 SKILLS BY OCCUPATION")
print("=" * 70)

for occupation in SELECTED_OCCUPATIONS:

    group = detailed_output[
        detailed_output["occupation_label"]
        .str.lower()
        == occupation
    ].head(10)

    print(f"\n{occupation.upper()}")

    print(
        group[
            [
                "skill_rank",
                "skill_label",
                "skill_posting_count",
                "skill_prevalence_percent",
                "demand_category",
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"Selected occupations: "
    f"{detailed_output['occupation_label'].nunique()}"
)

print(
    f"Occupation-skill rows: "
    f"{len(detailed_output)}"
)

print(
    f"Output saved to:\n{OUTPUT_FILE}"
)

print(
    f"\nSummary saved to:\n{SUMMARY_FILE}"
)

print("\nPhase 3 completed successfully.")