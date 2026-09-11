from pathlib import Path
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"



def build_skill_dataset():

    print("Loading processed datasets...")

    relations = pd.read_csv(
        PROCESSED_DIR / "esco_occupation_skill_relations.csv",
        low_memory=False
    )

    occupations = pd.read_csv(
        PROCESSED_DIR / "esco_occupations.csv",
        low_memory=False
    )

    skills = pd.read_csv(
        PROCESSED_DIR / "esco_skills.csv",
        low_memory=False
    )

    postings = pd.read_csv(
        PROCESSED_DIR / "job_postings_clean.csv",
        low_memory=False
    )

    posting_skills = pd.read_csv(
        PROCESSED_DIR / "job_posting_skills_clean.csv",
        low_memory=False
    )

    posting_titles = pd.read_csv(
        PROCESSED_DIR / "job_posting_titles_clean.csv",
        low_memory=False
    )


    relations["occupation_uri"] = (
        relations["occupation_uri"].astype("string").str.strip()
    )

    relations["skill_uri"] = (
        relations["skill_uri"].astype("string").str.strip()
    )

    occupations["occupation_uri"] = (
        occupations["occupation_uri"].astype("string").str.strip()
    )

    skills["skill_uri"] = (
        skills["skill_uri"].astype("string").str.strip()
    )

    postings["id"] = postings["id"].astype("string").str.strip()

    posting_skills["jobposting_id"] = (
        posting_skills["jobposting_id"].astype("string").str.strip()
    )

    posting_titles["jobposting_id"] = (
        posting_titles["jobposting_id"].astype("string").str.strip()
    )


    occupation_skill = relations.merge(
        occupations[
            [
                "occupation_uri",
                "occupation_label",
                "occupation_description"
            ]
        ],
        on="occupation_uri",
        how="left",
        suffixes=("", "_occupation")
    )

    occupation_skill = occupation_skill.merge(
        skills[
            [
                "skill_uri",
                "skill_label",
                "skill_type",
                "reuse_level",
                "skill_description"
            ]
        ],
        on="skill_uri",
        how="left",
        suffixes=("", "_skill")
    )

    occupation_skill = occupation_skill[
        [
            "occupation_uri",
            "occupation_label",
            "relation_type",
            "skill_uri",
            "skill_label",
            "skill_type",
            "reuse_level",
            "occupation_description",
            "skill_description"
        ]
    ]

    occupation_skill = occupation_skill.drop_duplicates()


    job_title_mapping = posting_titles[
        [
            "jobposting_id",
            "original_title",
            "title_esco_uri",
            "title_esco_label",
            "title_esco_fit_score"
        ]
    ].copy()

    job_title_mapping = job_title_mapping.rename(
        columns={
            "title_esco_uri": "occupation_uri",
            "title_esco_label": "occupation_label"
        }
    )


    posting_occupation = postings.merge(
        job_title_mapping[
            [
                "jobposting_id",
                "original_title",
                "occupation_uri",
                "occupation_label",
                "title_esco_fit_score"
            ]
        ],
        left_on="id",
        right_on="jobposting_id",
        how="left"
    )


    posting_skill_mapping = posting_skills[
        [
            "jobposting_id",
            "original_sentence",
            "skill_esco_uri",
            "skill_esco_label",
            "skill_esco_fit_score"
        ]
    ].copy()

    posting_skill_mapping = posting_skill_mapping.rename(
        columns={
            "skill_esco_uri": "skill_uri",
            "skill_esco_label": "skill_label"
        }
    )

    posting_skill_mapping = posting_occupation.merge(
        posting_skill_mapping,
        left_on="id",
        right_on="jobposting_id",
        how="left"
    )

    posting_skill_mapping = posting_skill_mapping.drop_duplicates()


    skill_demand = (
        posting_skill_mapping
        .dropna(subset=["occupation_uri", "skill_uri"])
        .groupby(
            [
                "occupation_uri",
                "occupation_label",
                "skill_uri",
                "skill_label"
            ],
            as_index=False
        )
        .agg(
            posting_count=("id", "nunique"),
            total_skill_mentions=("skill_uri", "size"),
            average_skill_fit_score=(
                "skill_esco_fit_score",
                "mean"
            )
        )
    )


    occupation_totals = (
        skill_demand
        .groupby("occupation_uri")["posting_count"]
        .transform("sum")
    )

    skill_demand["demand_share"] = (
        skill_demand["posting_count"] /
        occupation_totals.replace(0, pd.NA)
    )


    output_files = {
        "esco_occupation_skill_dataset.csv": occupation_skill,
        "job_posting_occupation_mapping.csv": posting_occupation,
        "job_posting_skill_mapping.csv": posting_skill_mapping,
        "skill_demand_by_occupation.csv": skill_demand
    }

    for filename, dataframe in output_files.items():
        output_path = PROCESSED_DIR / filename
        dataframe.to_csv(output_path, index=False)

        print(
            f"Saved {filename}: "
            f"{dataframe.shape[0]} rows, "
            f"{dataframe.shape[1]} columns"
        )

    print("\nSkill dataset construction completed.")


if __name__ == "__main__":
    build_skill_dataset()