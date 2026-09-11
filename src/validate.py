from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def safe_percentage(numerator, denominator):
    if denominator == 0:
        return 0.0
    return (numerator / denominator) * 100


# --------------------------------------------------
# Main validation
# --------------------------------------------------

def main():

    print_section("INTEGRATED DATA VALIDATION")

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    print("Loading processed datasets...")

    postings = pd.read_csv(
        DATA_DIR / "job_postings_clean.csv",
        low_memory=False
    )

    posting_occupations = pd.read_csv(
        DATA_DIR / "job_posting_occupation_mapping.csv",
        low_memory=False
    )

    posting_skills = pd.read_csv(
        DATA_DIR / "job_posting_skill_mapping.csv",
        low_memory=False
    )

    skill_demand = pd.read_csv(
        DATA_DIR / "skill_demand_by_occupation.csv",
        low_memory=False
    )

    esco_relations = pd.read_csv(
        DATA_DIR / "esco_occupation_skill_dataset.csv",
        low_memory=False
    )

    print("All datasets loaded successfully.")

    # --------------------------------------------------
    # Basic dataset sizes
    # --------------------------------------------------

    print_section("1. DATASET SIZES")

    datasets = {
        "Job postings": postings,
        "Posting → occupation": posting_occupations,
        "Posting → skill": posting_skills,
        "Skill demand": skill_demand,
        "ESCO occupation → skill": esco_relations,
    }

    for name, df in datasets.items():
        print(
            f"{name:<30} "
            f"Rows: {len(df):>8,} | "
            f"Columns: {len(df.columns):>3}"
        )

    # --------------------------------------------------
    # Job-posting occupation coverage
    # --------------------------------------------------

    print_section("2. JOB POSTING → ESCO OCCUPATION COVERAGE")

    total_postings = postings["id"].nunique()

    mapped_postings = (
        posting_occupations[
            posting_occupations["occupation_uri"].notna()
        ]["id"]
        .nunique()
    )

    unmapped_postings = total_postings - mapped_postings

    print(f"Total unique job postings:       {total_postings:,}")
    print(f"Mapped to ESCO occupation:        {mapped_postings:,}")
    print(f"Not mapped to ESCO occupation:    {unmapped_postings:,}")

    print(
        f"Occupation mapping coverage:     "
        f"{safe_percentage(mapped_postings, total_postings):.2f}%"
    )

    # --------------------------------------------------
    # Skill coverage
    # --------------------------------------------------

    print_section("3. JOB POSTING → SKILL COVERAGE")

    postings_with_skills = (
        posting_skills[
            posting_skills["skill_uri"].notna()
        ]["id"]
        .nunique()
    )

    postings_without_skills = (
        total_postings - postings_with_skills
    )

    print(f"Postings with extracted skills:   {postings_with_skills:,}")
    print(f"Postings without extracted skills:{postings_without_skills:,}")

    print(
        f"Skill coverage:                   "
        f"{safe_percentage(postings_with_skills, total_postings):.2f}%"
    )

    # --------------------------------------------------
    # Unique entities
    # --------------------------------------------------

    print_section("4. UNIQUE ENTITIES")

    unique_occupations = (
        posting_occupations["occupation_uri"]
        .dropna()
        .nunique()
    )

    unique_skills = (
        posting_skills["skill_uri"]
        .dropna()
        .nunique()
    )

    esco_occupations = (
        esco_relations["occupation_uri"]
        .dropna()
        .nunique()
    )

    esco_skills = (
        esco_relations["skill_uri"]
        .dropna()
        .nunique()
    )

    print(f"ESCO occupations in relations:   {esco_occupations:,}")
    print(f"ESCO skills in relations:         {esco_skills:,}")
    print(f"Occupations observed in jobs:     {unique_occupations:,}")
    print(f"Skills observed in job postings:  {unique_skills:,}")

    # --------------------------------------------------
    # Top occupations
    # --------------------------------------------------

    print_section("5. TOP ESCO OCCUPATIONS BY JOB POSTINGS")

    occupation_counts = (
        posting_occupations
        .dropna(subset=["occupation_uri"])
        .groupby(
            ["occupation_uri", "occupation_label"]
        )["id"]
        .nunique()
        .reset_index(name="posting_count")
        .sort_values(
            "posting_count",
            ascending=False
        )
        .head(20)
    )

    print(
        occupation_counts.to_string(index=False)
    )

    # --------------------------------------------------
    # Top skills
    # --------------------------------------------------

    print_section("6. TOP SKILLS BY JOB POSTING FREQUENCY")

    skill_counts = (
        posting_skills
        .dropna(subset=["skill_uri"])
        .groupby(
            ["skill_uri", "skill_label"]
        )["id"]
        .nunique()
        .reset_index(name="posting_count")
        .sort_values(
            "posting_count",
            ascending=False
        )
        .head(30)
    )

    print(
        skill_counts.to_string(index=False)
    )

    # --------------------------------------------------
    # Skill demand dataset
    # --------------------------------------------------

    print_section("7. SKILL DEMAND DATASET")

    print(
        f"Occupation-skill combinations:   "
        f"{len(skill_demand):,}"
    )

    print(
        f"Unique occupations:               "
        f"{skill_demand['occupation_uri'].nunique():,}"
    )

    print(
        f"Unique skills:                    "
        f"{skill_demand['skill_uri'].nunique():,}"
    )

    print(
        f"Total posting count represented:  "
        f"{skill_demand['posting_count'].sum():,}"
    )

    # --------------------------------------------------
    # Fit score analysis
    # --------------------------------------------------

    print_section("8. ESCO FIT SCORE ANALYSIS")

    if "title_esco_fit_score" in posting_occupations.columns:

        title_scores = pd.to_numeric(
            posting_occupations["title_esco_fit_score"],
            errors="coerce"
        ).dropna()

        print(f"Valid title fit scores:           {len(title_scores):,}")

        if len(title_scores) > 0:
            print(
                f"Mean:                              "
                f"{title_scores.mean():.4f}"
            )
            print(
                f"Median:                            "
                f"{title_scores.median():.4f}"
            )
            print(
                f"Minimum:                           "
                f"{title_scores.min():.4f}"
            )
            print(
                f"Maximum:                           "
                f"{title_scores.max():.4f}"
            )

    if "skill_esco_fit_score" in posting_skills.columns:

        skill_scores = pd.to_numeric(
            posting_skills["skill_esco_fit_score"],
            errors="coerce"
        ).dropna()

        print(f"\nValid skill fit scores:            {len(skill_scores):,}")

        if len(skill_scores) > 0:
            print(
                f"Mean:                              "
                f"{skill_scores.mean():.4f}"
            )
            print(
                f"Median:                            "
                f"{skill_scores.median():.4f}"
            )
            print(
                f"Minimum:                           "
                f"{skill_scores.min():.4f}"
            )
            print(
                f"Maximum:                           "
                f"{skill_scores.max():.4f}"
            )

    # --------------------------------------------------
    # Duplicate checks
    # --------------------------------------------------

    print_section("9. DUPLICATE CHECKS")

    print(
        f"Duplicate job posting IDs:        "
        f"{postings['id'].duplicated().sum():,}"
    )

    print(
        f"Duplicate posting-skill rows:      "
        f"{posting_skills.duplicated().sum():,}"
    )

    print(
        f"Duplicate occupation-skill rows:   "
        f"{esco_relations.duplicated().sum():,}"
    )

    # --------------------------------------------------
    # Missing critical identifiers
    # --------------------------------------------------

    print_section("10. CRITICAL IDENTIFIER CHECK")

    checks = {
        "Postings missing ID":
            postings["id"].isna().sum(),

        "Posting occupation rows missing posting ID":
            posting_occupations["id"].isna().sum(),

        "Posting skill rows missing posting ID":
            posting_skills["id"].isna().sum(),

        "Posting occupation rows missing occupation URI":
            posting_occupations["occupation_uri"].isna().sum(),

        "Posting skill rows missing skill URI":
            posting_skills["skill_uri"].isna().sum(),

        "Skill demand rows missing occupation URI":
            skill_demand["occupation_uri"].isna().sum(),

        "Skill demand rows missing skill URI":
            skill_demand["skill_uri"].isna().sum(),
    }

    for name, value in checks.items():
        print(f"{name:<55}: {value:,}")

    # --------------------------------------------------
    # Demand distribution
    # --------------------------------------------------

    print_section("11. SKILL DEMAND DISTRIBUTION")

    print(
        skill_demand["posting_count"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    print_section("12. VALIDATION COMPLETE")

    print(
        "Integrated data validation completed successfully."
    )

    print(
        "\nReview the results above before proceeding "
        "to skill-gap modelling."
    )


if __name__ == "__main__":
    main()