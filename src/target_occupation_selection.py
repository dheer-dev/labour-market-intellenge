"""
Target Occupation Selection
============================

Phase 2:
Identify and evaluate occupation candidates for occupation-specific
skill-demand and skill-gap analysis.

Stage A:
- Apply the frozen minimum-posting threshold.
- Analyse occupation-level skill diversity and sparsity.

Stage B:
- Build posting-level skill profiles using only postings with
  at least one valid extracted skill.
- Measure skill-profile stability.
- Analyse posting-title concentration and ESCO title-fit quality.

Outputs:
    data/processed/target_occupation_analysis.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MATRIX_FILE = (
    PROCESSED_DIR /
    "occupation_skill_demand_matrix.csv"
)

POSTING_SKILL_FILE = (
    PROCESSED_DIR /
    "job_posting_skill_mapping.csv"
)

POSTING_OCCUPATION_FILE = (
    PROCESSED_DIR /
    "job_posting_occupation_mapping.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR /
    "target_occupation_analysis.csv"
)

MIN_POSTINGS = 20

# Prevalence threshold used for Stage A descriptive analysis.
COMMON_SKILL_THRESHOLD = 0.20

# ESCO title-fit thresholds used for Stage B.
TITLE_FIT_80 = 80
TITLE_FIT_90 = 90


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def mean_pairwise_jaccard(skill_sets):
    """
    Calculate mean pairwise Jaccard similarity between posting
    skill sets.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Returns NaN if fewer than two profiles exist.
    """

    n = len(skill_sets)

    if n < 2:
        return np.nan

    similarities = []

    for i in range(n):
        set_a = skill_sets[i]

        for j in range(i + 1, n):
            set_b = skill_sets[j]

            union = set_a | set_b

            if not union:
                continue

            intersection = set_a & set_b

            similarities.append(
                len(intersection) / len(union)
            )

    if not similarities:
        return np.nan

    return float(np.mean(similarities))


def calculate_duplicate_profile_rate(skill_sets):
    """
    Calculate the percentage of postings whose complete skill
    profile is shared by at least one other posting.

    Example:
        A, A, B, C, C

    A occurs twice and C occurs twice.
    Four of five postings belong to duplicated profiles.

    Duplicate rate = 80%.
    """

    if not skill_sets:
        return np.nan

    profile_keys = [
        tuple(sorted(skill_set))
        for skill_set in skill_sets
    ]

    profile_counts = pd.Series(profile_keys).value_counts()

    duplicated_profiles = set(
        profile_counts[
            profile_counts > 1
        ].index
    )

    duplicated_postings = sum(
        profile in duplicated_profiles
        for profile in profile_keys
    )

    return (
        duplicated_postings
        / len(profile_keys)
        * 100
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TARGET OCCUPATION SELECTION")
print("=" * 70)

print("\nLoading processed datasets...")

matrix_df = pd.read_csv(MATRIX_FILE)
posting_skill_df = pd.read_csv(POSTING_SKILL_FILE)
posting_occupation_df = pd.read_csv(
    POSTING_OCCUPATION_FILE
)

print(
    f"Occupation-skill matrix shape: "
    f"{matrix_df.shape}"
)

print(
    f"Posting-skill mapping shape: "
    f"{posting_skill_df.shape}"
)

print(
    f"Posting-occupation mapping shape: "
    f"{posting_occupation_df.shape}"
)


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

matrix_required = {
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
}

posting_skill_required = {
    "jobposting_id_y",
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_esco_fit_score",
}

posting_occupation_required = {
    "jobposting_id_y",
    "occupation_uri",
    "occupation_label",
    "original_title",
    "title_esco_fit_score_y",
}

missing_matrix = (
    matrix_required
    - set(matrix_df.columns)
)

missing_posting_skill = (
    posting_skill_required
    - set(posting_skill_df.columns)
)

missing_posting_occupation = (
    posting_occupation_required
    - set(posting_occupation_df.columns)
)

if missing_matrix:
    raise ValueError(
        "Missing columns in occupation-skill matrix: "
        f"{sorted(missing_matrix)}"
    )

if missing_posting_skill:
    raise ValueError(
        "Missing columns in posting-skill mapping: "
        f"{sorted(missing_posting_skill)}"
    )

if missing_posting_occupation:
    raise ValueError(
        "Missing columns in posting-occupation mapping: "
        f"{sorted(missing_posting_occupation)}"
    )


# ============================================================
# STAGE A
# OCCUPATION-LEVEL SKILL PROFILE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STAGE A — OCCUPATION-LEVEL ANALYSIS")
print("=" * 70)


# ------------------------------------------------------------
# Validate occupation-skill uniqueness
# ------------------------------------------------------------

duplicate_matrix_rows = (
    matrix_df[
        [
            "occupation_uri",
            "skill_uri",
        ]
    ]
    .duplicated()
    .sum()
)

print(
    f"\nDuplicate occupation-skill rows: "
    f"{duplicate_matrix_rows}"
)

if duplicate_matrix_rows > 0:
    raise ValueError(
        "Occupation-skill matrix contains duplicate "
        "occupation-skill pairs."
    )


# ------------------------------------------------------------
# Identify occupations meeting minimum posting threshold
# ------------------------------------------------------------

occupation_summary = (
    matrix_df[
        [
            "occupation_uri",
            "occupation_label",
            "postings_with_skills",
        ]
    ]
    .drop_duplicates()
)

eligible = occupation_summary[
    occupation_summary["postings_with_skills"]
    >= MIN_POSTINGS
].copy()

print(
    f"\nMinimum posting threshold: "
    f"{MIN_POSTINGS}"
)

print(
    f"Eligible occupations: "
    f"{len(eligible)}"
)


# ------------------------------------------------------------
# Calculate Stage A occupation-level statistics
# ------------------------------------------------------------

stage_a_results = []

for _, occupation in eligible.iterrows():

    occupation_uri = occupation["occupation_uri"]
    occupation_label = occupation["occupation_label"]
    postings_with_skills = int(
        occupation["postings_with_skills"]
    )

    occupation_skills = matrix_df[
        matrix_df["occupation_uri"]
        == occupation_uri
    ].copy()

    distinct_observed_skills = (
        occupation_skills["skill_uri"]
        .nunique()
    )

    common_skill_count = (
        occupation_skills[
            occupation_skills["skill_prevalence"]
            >= COMMON_SKILL_THRESHOLD
        ]["skill_uri"]
        .nunique()
    )

    singleton_skill_count = (
        occupation_skills[
            occupation_skills["skill_posting_count"]
            == 1
        ]["skill_uri"]
        .nunique()
    )

    singleton_skill_share = (
        singleton_skill_count
        / distinct_observed_skills
        * 100
        if distinct_observed_skills > 0
        else np.nan
    )

    stage_a_results.append(
        {
            "occupation_uri": occupation_uri,
            "occupation_label": occupation_label,
            "postings_with_skills": postings_with_skills,
            "distinct_observed_skills": (
                distinct_observed_skills
            ),
            "common_skill_count": (
                common_skill_count
            ),
            "singleton_skill_count": (
                singleton_skill_count
            ),
            "singleton_skill_share_percent": (
                singleton_skill_share
            ),
        }
    )


stage_a_df = pd.DataFrame(stage_a_results)


print("\nStage A results:")

print(
    stage_a_df[
        [
            "occupation_label",
            "postings_with_skills",
            "distinct_observed_skills",
            "common_skill_count",
            "singleton_skill_count",
            "singleton_skill_share_percent",
        ]
    ]
    .sort_values(
        "postings_with_skills",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# STAGE B
# POSTING-LEVEL SKILL PROFILE STABILITY
# ============================================================

print("\n" + "=" * 70)
print("STAGE B — POSTING-LEVEL PROFILE STABILITY")
print("=" * 70)


# ------------------------------------------------------------
# IMPORTANT:
# Only retain rows with an actual extracted skill.
#
# The raw posting-skill mapping contains postings that have
# an occupation mapping but no extracted skill. These must NOT
# become posting-level skill profiles.
# ------------------------------------------------------------

posting_skill_rows = (
    posting_skill_df[
        [
            "jobposting_id_y",
            "occupation_uri",
            "occupation_label",
            "skill_uri",
        ]
    ]
    .dropna(subset=["skill_uri"])
    .drop_duplicates()
)


print(
    f"\nUnique posting-skill rows with valid skills: "
    f"{len(posting_skill_rows)}"
)


# ------------------------------------------------------------
# Create exactly one skill profile per posting
# ------------------------------------------------------------

posting_profiles = (
    posting_skill_rows
    .groupby(
        [
            "jobposting_id_y",
            "occupation_uri",
            "occupation_label",
        ]
    )["skill_uri"]
    .apply(set)
    .reset_index(name="skill_set")
)


print(
    f"Posting-level skill profiles created: "
    f"{len(posting_profiles)}"
)


# ------------------------------------------------------------
# Confirm Stage B posting counts match Stage A
# ------------------------------------------------------------

stage_b_counts = (
    posting_profiles
    .groupby(
        [
            "occupation_uri",
            "occupation_label",
        ]
    )["jobposting_id_y"]
    .nunique()
    .reset_index(
        name="stage_b_posting_count"
    )
)


posting_reconciliation = (
    stage_a_df[
        [
            "occupation_uri",
            "occupation_label",
            "postings_with_skills",
        ]
    ]
    .merge(
        stage_b_counts,
        on=[
            "occupation_uri",
            "occupation_label",
        ],
        how="left",
    )
)

posting_reconciliation[
    "stage_b_posting_count"
] = posting_reconciliation[
    "stage_b_posting_count"
].fillna(0).astype(int)

posting_reconciliation["difference"] = (
    posting_reconciliation["stage_b_posting_count"]
    - posting_reconciliation["postings_with_skills"]
)


print("\nPosting-count reconciliation:")

print(
    posting_reconciliation[
        [
            "occupation_label",
            "postings_with_skills",
            "stage_b_posting_count",
            "difference",
        ]
    ]
    .to_string(index=False)
)


if (
    posting_reconciliation["difference"]
    != 0
).any():

    raise ValueError(
        "\nStage A and Stage B posting counts do not "
        "match. Stage B should only contain postings "
        "with at least one extracted skill."
    )

print(
    "\n✓ Stage A and Stage B posting counts match."
)


# ------------------------------------------------------------
# Calculate profile stability
# ------------------------------------------------------------

stability_results = []

for _, occupation in eligible.iterrows():

    occupation_uri = occupation["occupation_uri"]
    occupation_label = occupation["occupation_label"]

    occupation_profiles = posting_profiles[
        posting_profiles["occupation_uri"]
        == occupation_uri
    ]

    skill_sets = (
        occupation_profiles["skill_set"]
        .tolist()
    )

    posting_count = len(skill_sets)

    unique_skill_profiles = len(
        {
            tuple(sorted(skill_set))
            for skill_set in skill_sets
        }
    )

    duplicate_rate = (
        calculate_duplicate_profile_rate(
            skill_sets
        )
    )

    mean_jaccard = (
        mean_pairwise_jaccard(
            skill_sets
        )
    )

    stability_results.append(
        {
            "occupation_uri": occupation_uri,
            "occupation_label": occupation_label,
            "posting_count_profiles": (
                posting_count
            ),
            "unique_skill_profiles": (
                unique_skill_profiles
            ),
            "duplicate_profile_rate_percent": (
                duplicate_rate
            ),
            "mean_pairwise_jaccard": (
                mean_jaccard
            ),
        }
    )


stability_df = pd.DataFrame(
    stability_results
)


print("\nPosting-level stability:")

print(
    stability_df[
        [
            "occupation_label",
            "posting_count_profiles",
            "unique_skill_profiles",
            "duplicate_profile_rate_percent",
            "mean_pairwise_jaccard",
        ]
    ]
    .sort_values(
        "posting_count_profiles",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# STAGE B
# TITLE CONCENTRATION AND TITLE-FIT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("STAGE B — TITLE / MAPPING QUALITY")
print("=" * 70)


# ------------------------------------------------------------
# One occupation-mapping row per posting
# ------------------------------------------------------------

# ------------------------------------------------------------
# Restrict title analysis to the same skill-bearing postings
# used in Stage A and Stage B.
# ------------------------------------------------------------

skill_bearing_posting_ids = set(
    posting_profiles["jobposting_id_y"]
)

posting_occupation_unique = (
    posting_occupation_df[
        [
            "jobposting_id_y",
            "occupation_uri",
            "occupation_label",
            "original_title",
            "title_esco_fit_score_y",
        ]
    ]
    .drop_duplicates(
        subset=["jobposting_id_y"]
    )
)

posting_occupation_unique = (
    posting_occupation_unique[
        posting_occupation_unique[
            "jobposting_id_y"
        ].isin(skill_bearing_posting_ids)
    ]
)


title_results = []

for _, occupation in eligible.iterrows():

    occupation_uri = occupation["occupation_uri"]
    occupation_label = occupation["occupation_label"]

    occupation_posts = posting_occupation_unique[
        posting_occupation_unique[
            "occupation_uri"
        ]
        == occupation_uri
    ].copy()

    posting_count = len(occupation_posts)

    if posting_count == 0:
        continue

    # --------------------------------------------------------
    # Title concentration
    # --------------------------------------------------------

    title_counts = (
        occupation_posts[
            "original_title"
        ]
        .fillna("UNKNOWN")
        .astype(str)
        .str.strip()
        .value_counts()
    )

    top_title = title_counts.index[0]

    top_title_count = int(
        title_counts.iloc[0]
    )

    top_title_share = (
        top_title_count
        / posting_count
        * 100
    )

    # --------------------------------------------------------
    # ESCO title-fit distribution
    # --------------------------------------------------------

    fit_scores = pd.to_numeric(
        occupation_posts[
            "title_esco_fit_score_y"
        ],
        errors="coerce",
    )

    fit_80_share = (
        (fit_scores >= TITLE_FIT_80).mean()
        * 100
    )

    fit_90_share = (
        (fit_scores >= TITLE_FIT_90).mean()
        * 100
    )

    median_fit = fit_scores.median()

    title_results.append(
        {
            "occupation_uri": occupation_uri,
            "occupation_label": occupation_label,
            "title_posting_count": posting_count,
            "top_posting_title": top_title,
            "top_posting_title_count": (
                top_title_count
            ),
            "top_posting_title_share_percent": (
                top_title_share
            ),
            "title_fit_ge_80_percent": (
                fit_80_share
            ),
            "title_fit_ge_90_percent": (
                fit_90_share
            ),
            "median_title_fit_score": (
                median_fit
            ),
        }
    )


title_df = pd.DataFrame(
    title_results
)


print("\nTitle concentration and title-fit:")

print(
    title_df[
        [
            "occupation_label",
            "title_posting_count",
            "top_posting_title",
            "top_posting_title_count",
            "top_posting_title_share_percent",
            "title_fit_ge_80_percent",
            "title_fit_ge_90_percent",
            "median_title_fit_score",
        ]
    ]
    .sort_values(
        "title_posting_count",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# MERGE ALL STAGE A + STAGE B RESULTS
# ============================================================

final_df = (
    stage_a_df
    .merge(
        stability_df,
        on=[
            "occupation_uri",
            "occupation_label",
        ],
        how="left",
    )
    .merge(
        title_df,
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
print("FINAL VALIDATION")
print("=" * 70)

print(
    f"\nFinal occupation count: "
    f"{len(final_df)}"
)

print(
    f"Final columns: "
    f"{len(final_df.columns)}"
)

if len(final_df) != len(eligible):
    raise ValueError(
        "Final result does not contain exactly one "
        "row per eligible occupation."
    )

if final_df[
    [
        "occupation_uri",
        "occupation_label",
    ]
].duplicated().any():

    raise ValueError(
        "Duplicate occupation rows found in final output."
    )

print(
    "\n✓ One row per threshold-qualified occupation."
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
    f"\nSaved final analysis to:\n"
    f"{OUTPUT_FILE}"
)

print("\n" + "=" * 70)
print("STAGE A + STAGE B COMPLETE")
print("=" * 70)