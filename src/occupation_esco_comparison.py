"""
Occupation–ESCO Comparison

==========================

Phase 2 — Stage C:

Compare threshold-qualified occupations against their ESCO
occupation-skill profiles.

Stage C measures:

1. ESCO skill vocabulary size
2. Observed market skill vocabulary size
3. Observed-to-ESCO skill coverage
4. ESCO-to-observed skill coverage
5. Skill overlap
6. Coverage of common market skills
7. Coverage of singleton market skills
8. ESCO essential / optional skill composition, where available

Important:
This stage does NOT calculate a final skill gap.

The purpose is to establish whether the observed job-market
skill profile and the ESCO occupational profile correspond
sufficiently well for later occupation-specific skill-gap analysis.

Input:
    data/processed/target_occupation_analysis.csv
    data/processed/esco_occupation_skill_relations.csv
    data/processed/esco_occupations.csv
    data/processed/esco_skills.csv

Output:
    data/processed/target_occupation_esco_analysis.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_OCCUPATION_FILE = (
    PROCESSED_DIR /
    "target_occupation_analysis.csv"
)

ESCO_RELATIONS_FILE = (
    PROCESSED_DIR /
    "esco_occupation_skill_relations.csv"
)

ESCO_OCCUPATIONS_FILE = (
    PROCESSED_DIR /
    "esco_occupations.csv"
)

ESCO_SKILLS_FILE = (
    PROCESSED_DIR /
    "esco_skills.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "target_occupation_esco_analysis.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentage(numerator, denominator):
    """
    Safely calculate a percentage.
    """
    if denominator == 0:
        return np.nan

    return numerator / denominator * 100


def normalize_uri(series):
    """
    Normalize URI columns for reliable joins.

    Removes surrounding whitespace and converts values to strings.
    """
    return (
        series
        .astype(str)
        .str.strip()
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STAGE C — ESCO OCCUPATION–SKILL COMPARISON")
print("=" * 70)

print("\nLoading processed datasets...")

target_df = pd.read_csv(
    TARGET_OCCUPATION_FILE
)

esco_relations_df = pd.read_csv(
    ESCO_RELATIONS_FILE
)

esco_occupations_df = pd.read_csv(
    ESCO_OCCUPATIONS_FILE
)

esco_skills_df = pd.read_csv(
    ESCO_SKILLS_FILE
)

print(
    f"Target occupation analysis shape: "
    f"{target_df.shape}"
)

print(
    f"ESCO occupation-skill relations shape: "
    f"{esco_relations_df.shape}"
)

print(
    f"ESCO occupations shape: "
    f"{esco_occupations_df.shape}"
)

print(
    f"ESCO skills shape: "
    f"{esco_skills_df.shape}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

target_required = {
    "occupation_uri",
    "occupation_label",
    "postings_with_skills",
    "distinct_observed_skills",
    "common_skill_count",
    "singleton_skill_count",
    "singleton_skill_share_percent",
}

esco_relation_required = {
    "occupation_uri",
    "skill_uri",
}

esco_occupation_required = {
    "occupation_uri",
    "occupation_label",
}

esco_skill_required = {
    "skill_uri",
    "skill_label",
}

missing_target = (
    target_required
    - set(target_df.columns)
)

missing_relations = (
    esco_relation_required
    - set(esco_relations_df.columns)
)

missing_occupations = (
    esco_occupation_required
    - set(esco_occupations_df.columns)
)

missing_skills = (
    esco_skill_required
    - set(esco_skills_df.columns)
)

if missing_target:
    raise ValueError(
        "Missing columns in target occupation analysis: "
        f"{sorted(missing_target)}"
    )

if missing_relations:
    raise ValueError(
        "Missing columns in ESCO occupation-skill relations: "
        f"{sorted(missing_relations)}"
    )

if missing_occupations:
    raise ValueError(
        "Missing columns in ESCO occupations: "
        f"{sorted(missing_occupations)}"
    )

if missing_skills:
    raise ValueError(
        "Missing columns in ESCO skills: "
        f"{sorted(missing_skills)}"
    )


# ============================================================
# NORMALIZE URI COLUMNS
# ============================================================

print("\nNormalizing ESCO URIs...")

target_df["occupation_uri"] = normalize_uri(
    target_df["occupation_uri"]
)

esco_relations_df["occupation_uri"] = normalize_uri(
    esco_relations_df["occupation_uri"]
)

esco_relations_df["skill_uri"] = normalize_uri(
    esco_relations_df["skill_uri"]
)

esco_occupations_df["occupation_uri"] = normalize_uri(
    esco_occupations_df["occupation_uri"]
)

esco_skills_df["skill_uri"] = normalize_uri(
    esco_skills_df["skill_uri"]
)


# ============================================================
# VALIDATE ESCO RELATION UNIQUENESS
# ============================================================

duplicate_relations = (
    esco_relations_df[
        [
            "occupation_uri",
            "skill_uri",
        ]
    ]
    .duplicated()
    .sum()
)

print(
    f"\nDuplicate ESCO occupation-skill relations: "
    f"{duplicate_relations}"
)

if duplicate_relations > 0:
    print(
        "WARNING: Duplicate ESCO occupation-skill "
        "relations detected."
    )

    esco_relations_df = (
        esco_relations_df
        .drop_duplicates(
            subset=[
                "occupation_uri",
                "skill_uri",
            ]
        )
    )


# ============================================================
# IDENTIFY TARGET OCCUPATIONS
# ============================================================

target_occupations = (
    target_df[
        [
            "occupation_uri",
            "occupation_label",
        ]
    ]
    .drop_duplicates()
)

print(
    f"\nTarget occupations entering Stage C: "
    f"{len(target_occupations)}"
)


# ============================================================
# BUILD ESCO SKILL PROFILES
# ============================================================

print("\nBuilding ESCO occupation skill profiles...")

target_occupation_uris = set(
    target_occupations["occupation_uri"]
)

esco_target_relations = (
    esco_relations_df[
        esco_relations_df["occupation_uri"]
        .isin(target_occupation_uris)
    ]
    .copy()
)

print(
    f"ESCO occupation-skill relations for target "
    f"occupations: {len(esco_target_relations)}"
)


# ============================================================
# CHECK ESCO OCCUPATION COVERAGE
# ============================================================

esco_occupation_coverage = (
    target_occupations[
        ["occupation_uri", "occupation_label"]
    ]
    .merge(
        esco_target_relations[
            ["occupation_uri"]
        ]
        .drop_duplicates(),
        on="occupation_uri",
        how="left",
        indicator=True,
    )
)

missing_esco_occupations = (
    esco_occupation_coverage[
        esco_occupation_coverage["_merge"]
        == "left_only"
    ]
)

if len(missing_esco_occupations) > 0:

    print(
        "\nWARNING:"
        f" {len(missing_esco_occupations)} target "
        "occupations have no ESCO skill relations."
    )

    print(
        missing_esco_occupations[
            [
                "occupation_uri",
                "occupation_label",
            ]
        ]
        .to_string(index=False)
    )


# ============================================================
# CREATE ESCO SKILL SET PER OCCUPATION
# ============================================================

esco_skill_sets = {}

for occupation_uri, group in (
    esco_target_relations
    .groupby("occupation_uri")
):

    esco_skill_sets[occupation_uri] = set(
        group["skill_uri"]
        .dropna()
    )


# ============================================================
# LOAD OBSERVED MARKET SKILLS
# ============================================================

print(
    "\nLoading observed market skill profiles..."
)

# Reconstruct occupation-level observed skill sets
# directly from the processed occupation-skill matrix.

OBSERVED_MATRIX_FILE = (
    PROCESSED_DIR /
    "occupation_skill_demand_matrix.csv"
)

observed_matrix_df = pd.read_csv(
    OBSERVED_MATRIX_FILE
)

observed_matrix_required = {
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_posting_count",
    "skill_prevalence",
}

missing_observed = (
    observed_matrix_required
    - set(observed_matrix_df.columns)
)

if missing_observed:
    raise ValueError(
        "Missing columns in occupation-skill matrix: "
        f"{sorted(missing_observed)}"
    )

observed_matrix_df["occupation_uri"] = normalize_uri(
    observed_matrix_df["occupation_uri"]
)

observed_matrix_df["skill_uri"] = normalize_uri(
    observed_matrix_df["skill_uri"]
)

observed_matrix_df = observed_matrix_df[
    observed_matrix_df["occupation_uri"]
    .isin(target_occupation_uris)
].copy()


# ============================================================
# BUILD OBSERVED SKILL SETS
# ============================================================

observed_skill_sets = {}

for occupation_uri, group in (
    observed_matrix_df
    .groupby("occupation_uri")
):

    observed_skill_sets[occupation_uri] = set(
        group["skill_uri"]
        .dropna()
    )


# ============================================================
# STAGE C COMPARISON
# ============================================================

print("\nCalculating ESCO comparison metrics...")

stage_c_results = []

for _, occupation in target_occupations.iterrows():

    occupation_uri = occupation[
        "occupation_uri"
    ]

    occupation_label = occupation[
        "occupation_label"
    ]

    # --------------------------------------------------------
    # Observed market skill set
    # --------------------------------------------------------

    observed_skills = observed_skill_sets.get(
        occupation_uri,
        set()
    )

    # --------------------------------------------------------
    # ESCO skill set
    # --------------------------------------------------------

    esco_skills = esco_skill_sets.get(
        occupation_uri,
        set()
    )

    # --------------------------------------------------------
    # Skill overlap
    # --------------------------------------------------------

    overlap = (
        observed_skills
        & esco_skills
    )

    observed_only = (
        observed_skills
        - esco_skills
    )

    esco_only = (
        esco_skills
        - observed_skills
    )

    # --------------------------------------------------------
    # Coverage metrics
    # --------------------------------------------------------

    observed_to_esco = percentage(
        len(overlap),
        len(observed_skills)
    )

    esco_to_observed = percentage(
        len(overlap),
        len(esco_skills)
    )

    # --------------------------------------------------------
    # Jaccard similarity
    # --------------------------------------------------------

    union = (
        observed_skills
        | esco_skills
    )

    if union:
        jaccard_similarity = (
            len(overlap)
            / len(union)
        )
    else:
        jaccard_similarity = np.nan

    # --------------------------------------------------------
    # Common / singleton observed skills
    # --------------------------------------------------------

    occupation_observed_rows = (
        observed_matrix_df[
            observed_matrix_df["occupation_uri"]
            == occupation_uri
        ]
    )

    common_observed_skills = set(
        occupation_observed_rows[
            occupation_observed_rows[
                "skill_prevalence"
            ] >= 0.20
        ]["skill_uri"]
    )

    singleton_observed_skills = set(
        occupation_observed_rows[
            occupation_observed_rows[
                "skill_posting_count"
            ] == 1
        ]["skill_uri"]
    )

    common_overlap = (
        common_observed_skills
        & esco_skills
    )

    singleton_overlap = (
        singleton_observed_skills
        & esco_skills
    )

    common_skill_coverage = percentage(
        len(common_overlap),
        len(common_observed_skills)
    )

    singleton_skill_coverage = percentage(
        len(singleton_overlap),
        len(singleton_observed_skills)
    )

    # --------------------------------------------------------
    # Retrieve ESCO occupation label
    # --------------------------------------------------------

    esco_occ_match = (
        esco_occupations_df[
            esco_occupations_df["occupation_uri"]
            == occupation_uri
        ]
    )

    if len(esco_occ_match) > 0:
        esco_label = (
            esco_occ_match.iloc[0][
                "occupation_label"
            ]
        )
    else:
        esco_label = occupation_label

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    stage_c_results.append(
        {
            "occupation_uri": occupation_uri,
            "occupation_label": occupation_label,
            "esco_preferred_label": esco_label,

            "observed_skill_count": (
                len(observed_skills)
            ),

            "esco_skill_count": (
                len(esco_skills)
            ),

            "observed_esco_overlap_count": (
                len(overlap)
            ),

            "observed_only_skill_count": (
                len(observed_only)
            ),

            "esco_only_skill_count": (
                len(esco_only)
            ),

            "observed_to_esco_coverage_percent": (
                observed_to_esco
            ),

            "esco_to_observed_coverage_percent": (
                esco_to_observed
            ),

            "observed_esco_jaccard": (
                jaccard_similarity
            ),

            "common_observed_skill_count": (
                len(common_observed_skills)
            ),

            "common_observed_esco_overlap_count": (
                len(common_overlap)
            ),

            "common_skill_coverage_percent": (
                common_skill_coverage
            ),

            "singleton_observed_skill_count": (
                len(singleton_observed_skills)
            ),

            "singleton_observed_esco_overlap_count": (
                len(singleton_overlap)
            ),

            "singleton_skill_coverage_percent": (
                singleton_skill_coverage
            ),
        }
    )


stage_c_df = pd.DataFrame(
    stage_c_results
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\nStage C — ESCO comparison results:"
)

display_columns = [
    "occupation_label",
    "observed_skill_count",
    "esco_skill_count",
    "observed_esco_overlap_count",
    "observed_to_esco_coverage_percent",
    "esco_to_observed_coverage_percent",
    "observed_esco_jaccard",
    "common_skill_coverage_percent",
    "singleton_skill_coverage_percent",
]

print(
    stage_c_df[
        display_columns
    ]
    .sort_values(
        "observed_esco_jaccard",
        ascending=False,
        na_position="last",
    )
    .to_string(index=False)
)


# ============================================================
# MERGE WITH STAGE A + STAGE B
# ============================================================

print(
    "\nMerging Stage A + Stage B + Stage C..."
)

final_df = (
    target_df
    .merge(
        stage_c_df,
        on=[
            "occupation_uri",
            "occupation_label",
        ],
        how="left",
    )
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STAGE C FINAL VALIDATION")
print("=" * 70)

print(
    f"\nFinal occupation count: "
    f"{len(final_df)}"
)

print(
    f"Final columns: "
    f"{len(final_df.columns)}"
)

if len(final_df) != len(target_df):

    raise ValueError(
        "Stage C merge changed the number of "
        "occupation rows."
    )

if final_df[
    [
        "occupation_uri",
        "occupation_label",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate occupation rows found "
        "after Stage C merge."
    )

print(
    "\n✓ Stage C preserved exactly one row "
    "per target occupation."
)


# ============================================================
# SAVE OUTPUT
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"\nSaved Stage C analysis to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("STAGE C COMPLETE")
print("=" * 70)