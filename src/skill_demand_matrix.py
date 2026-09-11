from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def main():

    print("=" * 70)
    print("BUILDING OCCUPATION-SKILL DEMAND MATRIX")
    print("=" * 70)

    df = pd.read_csv(
        DATA_DIR / "job_posting_skill_mapping.csv",
        low_memory=False
    )


    df = df.dropna(
        subset=["occupation_uri", "skill_uri"]
    ).copy()

    print(f"\nValid occupation-skill records: {len(df):,}")


    df = df.drop_duplicates(
        subset=[
            "id",
            "occupation_uri",
            "skill_uri"
        ]
    )

    print(
        f"Unique posting-occupation-skill records: "
        f"{len(df):,}"
    )


    occupation_postings = (
        df.groupby(
            ["occupation_uri", "occupation_label"]
        )["id"]
        .nunique()
        .reset_index(
            name="postings_with_skills"
        )
    )


    skill_counts = (
        df.groupby(
            [
                "occupation_uri",
                "occupation_label",
                "skill_uri",
                "skill_label"
            ])["id"]
        .nunique()
        .reset_index(
            name="skill_posting_count"
        )
    )


    demand = skill_counts.merge(
        occupation_postings,
        on=[
            "occupation_uri",
            "occupation_label"
        ],
        how="left"
    )


    demand["skill_prevalence"] = (
        demand["skill_posting_count"] /
        demand["postings_with_skills"]
    )

    demand["skill_prevalence_percent"] = (
        demand["skill_prevalence"] * 100
    )


    demand = demand.sort_values(
        [
            "occupation_label",
            "skill_prevalence"
        ],
        ascending=[True, False]
    )


    output_path = (
        DATA_DIR /
        "occupation_skill_demand_matrix.csv"
    )

    demand.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved: {output_path}"
    )

    print(
        f"Rows: {len(demand):,}"
    )

    print(
        f"Columns: {len(demand.columns)}"
    )


    print("\nTop occupations by number of postings with skills:")

    top_occupations = (
        occupation_postings
        .sort_values(
            "postings_with_skills",
            ascending=False
        )
        .head(20)
    )

    print(
        top_occupations.to_string(
            index=False
        )
    )


    print("\nMost frequently observed skills overall:")

    overall = (
        df.groupby(
            ["skill_uri", "skill_label"]
        )["id"]
        .nunique()
        .reset_index(
            name="posting_count"
        )
        .sort_values(
            "posting_count",
            ascending=False
        )
        .head(20)
    )

    print(
        overall.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("SKILL-DEMAND MATRIX COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()