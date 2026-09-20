"""Phase 1: occupation-level inspection of the skill-demand matrix.

Reads data/processed/occupation_skill_demand_matrix.csv and writes
occupation sample-size summaries. Does not alter the matrix.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MATRIX_PATH = DATA_DIR / "occupation_skill_demand_matrix.csv"


def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def occupation_summary(matrix):
    return (
        matrix.groupby(
            ["occupation_uri", "occupation_label"],
            as_index=False,
        )
        .agg(
            postings_with_skills=("postings_with_skills", "first"),
            n_distinct_skills=("skill_uri", "nunique"),
            median_skill_prevalence=("skill_prevalence", "median"),
            mean_skill_prevalence=("skill_prevalence", "mean"),
            max_skill_prevalence=("skill_prevalence", "max"),
            n_singleton_skills=("skill_posting_count", lambda s: (s == 1).sum()),
            n_skills_prevalence_ge_10=("skill_prevalence", lambda s: (s >= 0.10).sum()),
            n_skills_prevalence_ge_20=("skill_prevalence", lambda s: (s >= 0.20).sum()),
        )
        .sort_values("postings_with_skills", ascending=False)
    )


def threshold_table(occupations):
    rows = []
    for threshold in [5, 10, 15, 20, 25, 30, 40, 50]:
        subset = occupations[occupations["postings_with_skills"] >= threshold]
        rows.append(
            {
                "min_postings_with_skills": threshold,
                "n_occupations": len(subset),
                "median_postings": subset["postings_with_skills"].median(),
                "median_n_skills": subset["n_distinct_skills"].median(),
            }
        )
    return pd.DataFrame(rows)


def main():
    print_section("PHASE 1: OCCUPATION-LEVEL SKILL ANALYSIS")

    matrix = pd.read_csv(MATRIX_PATH, low_memory=False)
    print(f"Loaded: {MATRIX_PATH}")
    print(f"Rows: {len(matrix):,} | Columns: {len(matrix.columns)}")
    print(f"Columns: {list(matrix.columns)}")

    occupations = occupation_summary(matrix)

    print_section("1. MATRIX CONTENTS")
    print(f"Occupation-skill combinations: {len(matrix):,}")
    print(f"Unique occupations:            {occupations['occupation_uri'].nunique():,}")
    print(f"Unique skills:                 {matrix['skill_uri'].nunique():,}")
    print(
        "Duplicate occupation-skill rows: "
        f"{matrix.duplicated(['occupation_uri', 'skill_uri']).sum():,}"
    )

    print_section("2. POSTING-COUNT DISTRIBUTION (occupation level)")
    print(occupations["postings_with_skills"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    ).to_string())

    print_section("3. SKILL-COUNT DISTRIBUTION (occupation level)")
    print(occupations["n_distinct_skills"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    ).to_string())

    print_section("4. SKILL PREVALENCE DISTRIBUTION (matrix rows)")
    print(matrix["skill_prevalence"].describe(
        percentiles=[0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    ).to_string())
    print(
        "\nRows with skill_posting_count == 1: "
        f"{(matrix['skill_posting_count'] == 1).mean():.1%}"
    )
    print(
        "Rows with skill_posting_count <= 2: "
        f"{(matrix['skill_posting_count'] <= 2).mean():.1%}"
    )

    print_section("5. THRESHOLD SENSITIVITY")
    thresholds = threshold_table(occupations)
    print(thresholds.to_string(index=False))

    print_section("6. OCCUPATIONS WITH >= 20 POSTINGS WITH SKILLS")
    eligible = occupations[occupations["postings_with_skills"] >= 20]
    print(eligible.to_string(index=False))

    out_path = DATA_DIR / "occupation_sample_sizes.csv"
    occupations.to_csv(out_path, index=False)
    print_section("OUTPUT")
    print(f"Saved occupation sample sizes: {out_path}")
    print(f"Rows: {len(occupations):,}")


if __name__ == "__main__":
    main()
