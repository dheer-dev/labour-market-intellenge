from pathlib import Path
import pandas as pd



PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ESCO_DIR = RAW_DIR / "esco"
JOBS_DIR = RAW_DIR / "job_postings"
ONET_DIR = RAW_DIR / "onet"


def clean_column_names(df):
    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("*", "", regex=False)
        .str.replace("#", "number", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    return df


def clean_text_column(df, column):
    if column in df.columns:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )
    return df


def save_processed(df, filename):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Shape: {df.shape}")


def process_esco():
    print("\nProcessing ESCO data...")

    relations_path = ESCO_DIR / "occupationSkillRelations_en.csv"
    occupations_path = ESCO_DIR / "occupations_en.csv"
    skills_path = ESCO_DIR / "skills_en.csv"

    relations = pd.read_csv(relations_path, low_memory=False)
    occupations = pd.read_csv(occupations_path, low_memory=False)
    skills = pd.read_csv(skills_path, low_memory=False)

    relations = clean_column_names(relations)
    occupations = clean_column_names(occupations)
    skills = clean_column_names(skills)


    relations = relations.drop_duplicates(
        subset=[
            "occupationuri",
            "skilluri",
            "relationtype"
        ]
    )


    occupations_clean = occupations[
        [
            "concepturi",
            "preferredlabel",
            "altlabels",
            "status",
            "code",
            "description"
        ]
    ].copy()

    occupations_clean = occupations_clean.rename(
        columns={
            "concepturi": "occupation_uri",
            "preferredlabel": "occupation_label",
            "altlabels": "occupation_aliases",
            "description": "occupation_description"
        }
    )


    skills_clean = skills[
        [
            "concepturi",
            "preferredlabel",
            "altlabels",
            "skilltype",
            "reuselevel",
            "description"
        ]
    ].copy()

    skills_clean = skills_clean.rename(
        columns={
            "concepturi": "skill_uri",
            "preferredlabel": "skill_label",
            "altlabels": "skill_aliases",
            "skilltype": "skill_type",
            "reuselevel": "reuse_level",
            "description": "skill_description"
        }
    )

    for column in [
        "occupation_label",
        "occupation_aliases",
        "occupation_description"
    ]:
        clean_text_column(occupations_clean, column)

    for column in [
        "skill_label",
        "skill_aliases",
        "skill_description"
    ]:
        clean_text_column(skills_clean, column)


    relations = relations.rename(
        columns={
            "occupationuri": "occupation_uri",
            "occupationlabel": "occupation_label",
            "relationtype": "relation_type",
            "skilltype": "skill_type",
            "skilluri": "skill_uri",
            "skilllabel": "skill_label"
        }
    )

    save_processed(
        relations,
        "esco_occupation_skill_relations.csv"
    )

    save_processed(
        occupations_clean,
        "esco_occupations.csv"
    )

    save_processed(
        skills_clean,
        "esco_skills.csv"
    )


def process_job_postings():
    print("\nProcessing job-posting data...")

    postings_path = (JOBS_DIR / "NYC Jobs (snapshot 2025-07-13) (anonymized).xlsx")

    skills_path = JOBS_DIR / "skills - 15000 sample.xlsx"
    titles_path = JOBS_DIR / "titles.xlsx"

    postings = pd.read_excel(postings_path)
    posting_skills = pd.read_excel(skills_path)
    posting_titles = pd.read_excel(titles_path)

    postings = clean_column_names(postings)
    posting_skills = clean_column_names(posting_skills)
    posting_titles = clean_column_names(posting_titles)


    postings = postings.drop_duplicates(subset=["id"])
    posting_skills = posting_skills.drop_duplicates()
    posting_titles = posting_titles.drop_duplicates(
        subset=["jobposting_id"]
    )


    postings["id"] = postings["id"].astype("string")
    posting_skills["jobposting_id"] = (
        posting_skills["jobposting_id"].astype("string")
    )
    posting_titles["jobposting_id"] = (
        posting_titles["jobposting_id"].astype("string")
    )

    posting_text_columns = [
        "company_name",
        "title",
        "civil_service_title",
        "industry",
        "description",
        "minimum_qual_requirements",
        "additional_information"
    ]

    for column in posting_text_columns:
        clean_text_column(postings, column)

    for column in [
        "original_sentence",
        "skill_esco_label",
        "skill_esco_alias"
    ]:
        clean_text_column(posting_skills, column)

    for column in [
        "original_title",
        "title_esco_label",
        "title_esco_alias"
    ]:
        clean_text_column(posting_titles, column)


    for column in [
        "posting_date",
        "post_until",
        "posting_updated"
    ]:
        if column in postings.columns:
            postings[column] = pd.to_datetime(
                postings[column],
                errors="coerce"
            )


    for column in [
        "salary_range_from",
        "salary_range_to",
        "number_of_positions"
    ]:
        if column in postings.columns:
            postings[column] = pd.to_numeric(
                postings[column],
                errors="coerce"
            )


    postings["average_salary"] = (
        postings["salary_range_from"] +
        postings["salary_range_to"]
    ) / 2


    postings_with_titles = postings.merge(
        posting_titles[
            [
                "jobposting_id",
                "title_esco_uri",
                "title_esco_label",
                "title_esco_fit_score"
            ]
        ],
        left_on="id",
        right_on="jobposting_id",
        how="left"
    )

    save_processed(
        postings_with_titles,
        "job_postings_clean.csv"
    )

    save_processed(
        posting_skills,
        "job_posting_skills_clean.csv"
    )

    save_processed(
        posting_titles,
        "job_posting_titles_clean.csv"
    )


def process_onet():
    print("\nProcessing O*NET data...")

    essential_path = ONET_DIR / "essential_skills.csv"
    transferable_path = ONET_DIR / "transferable_skills.csv"
    knowledge_path = ONET_DIR / "knowledge.csv"
    occupation_path = ONET_DIR / "occupation_data.csv"
    titles_path = ONET_DIR / "job_titles.csv"

    essential = pd.read_csv(essential_path, low_memory=False)
    transferable = pd.read_csv(
        transferable_path,
        low_memory=False
    )
    knowledge = pd.read_csv(knowledge_path, low_memory=False)
    occupations = pd.read_csv(
        occupation_path,
        low_memory=False
    )
    titles = pd.read_csv(titles_path, low_memory=False)

    essential = clean_column_names(essential)
    transferable = clean_column_names(transferable)
    knowledge = clean_column_names(knowledge)
    occupations = clean_column_names(occupations)
    titles = clean_column_names(titles)


    essential = essential.drop_duplicates()
    transferable = transferable.drop_duplicates()
    knowledge = knowledge.drop_duplicates()
    occupations = occupations.drop_duplicates()
    titles = titles.drop_duplicates()


    for df in [essential, transferable, knowledge]:
        if "data_value" in df.columns:
            df["data_value"] = pd.to_numeric(
                df["data_value"],
                errors="coerce"
            )

    for df in [
        essential,
        transferable,
        knowledge,
        occupations,
        titles
    ]:
        for column in ["title", "element_name", "job_title", "description"]:
            clean_text_column(df, column)

    save_processed(
        essential,
        "onet_essential_skills_clean.csv"
    )

    save_processed(
        transferable,
        "onet_transferable_skills_clean.csv"
    )

    save_processed(
        knowledge,
        "onet_knowledge_clean.csv"
    )

    save_processed(
        occupations,
        "onet_occupations_clean.csv"
    )

    save_processed(
        titles,
        "onet_job_titles_clean.csv"
    )



def main():
    print("Starting preprocessing pipeline...")

    process_esco()
    process_job_postings()
    process_onet()

    print("\nPreprocessing completed successfully.")


if __name__ == "__main__":
    main()