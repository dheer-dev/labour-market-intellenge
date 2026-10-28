
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

MAPPING_FILE = PROCESSED / "onet_esco_occupation_mapping_reviewed.csv"

INPUT_FILES = {
    "Essential Skills": PROCESSED / "onet_essential_skills_clean.csv",
    "Transferable Skills": PROCESSED / "onet_transferable_skills_clean.csv",
}

OUTPUT_FILE = PROCESSED / "onet_target_occupation_skills.csv"


def main():
    mapping = pd.read_csv(
        MAPPING_FILE,
        dtype={"onet_soc_code": str},
    )

    approved = mapping[
        mapping["review_status"].eq("APPROVED")
    ].copy()

    if approved.empty:
        raise ValueError("No approved occupation mappings were found.")

    # Each approved target must map to one O*NET code in this first pass.
    if approved["target_occupation"].duplicated().any():
        duplicates = approved.loc[
            approved["target_occupation"].duplicated(keep=False),
            ["target_occupation", "onet_soc_code"],
        ]
        raise ValueError(
            "Multiple approved O*NET mappings exist for a target. "
            "Review these mappings before proceeding:\n"
            + duplicates.to_string(index=False)
        )

    mapping_columns = [
        "target_occupation",
        "onet_soc_code",
        "onet_occupation_title",
    ]
    approved = approved[mapping_columns].copy()

    all_categories = []

    for category, path in INPUT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")

        df = pd.read_csv(
            path,
            dtype={"onet_soc_code": str},
        )

        required = {
            "onet_soc_code",
            "title",
            "element_id",
            "element_name",
            "scale_name",
            "data_value",
        }
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{path.name} is missing columns: {sorted(missing)}"
            )

        df = df[
            df["onet_soc_code"].isin(approved["onet_soc_code"])
        ].copy()

        if df.empty:
            print(f"WARNING: No matching records found in {path.name}")
            continue

        df["data_value"] = pd.to_numeric(
            df["data_value"], errors="raise"
        )

        unexpected_scales = (
            set(df["scale_name"].dropna().unique())
            - {"Importance", "Level"}
        )
        if unexpected_scales:
            raise ValueError(
                f"Unexpected scale names in {path.name}: "
                f"{sorted(unexpected_scales)}"
            )

        keys = [
            "onet_soc_code",
            "title",
            "element_id",
            "element_name",
        ]

        duplicate_mask = df.duplicated(
            subset=keys + ["scale_name"],
            keep=False,
        )

        if duplicate_mask.any():
            examples = df.loc[
                duplicate_mask, keys + ["scale_name", "data_value"]
            ].head(10)

            raise ValueError(
                f"Duplicate occupation-skill-scale records in "
                f"{path.name}. Inspect before aggregation:\n"
                f"{examples.to_string(index=False)}"
            )

        wide = df.pivot(
            index=keys,
            columns="scale_name",
            values="data_value",
        ).reset_index()

        wide.columns.name = None

        for scale in ["Importance", "Level"]:
            if scale not in wide.columns:
                wide[scale] = pd.NA

        wide = wide.rename(
            columns={
                "Importance": "onet_importance",
                "Level": "onet_level",
            }
        )

        wide["onet_skill_category"] = category
        wide["source_dataset"] = path.name

        all_categories.append(wide)

    if not all_categories:
        raise ValueError(
            "No O*NET skill records matched the approved mappings."
        )

    skills = pd.concat(all_categories, ignore_index=True)

    skills = skills.merge(
        approved,
        on="onet_soc_code",
        how="left",
        validate="many_to_one",
        suffixes=("", "_mapping"),
    )

    if skills["target_occupation"].isna().any():
        raise ValueError(
            "Some O*NET skill records could not be linked to an "
            "approved target occupation."
        )

    output_columns = [
        "target_occupation",
        "onet_soc_code",
        "onet_occupation_title",
        "title",
        "element_id",
        "element_name",
        "onet_skill_category",
        "onet_importance",
        "onet_level",
        "source_dataset",
    ]

    skills = skills[output_columns].sort_values(
        [
            "target_occupation",
            "onet_skill_category",
            "onet_importance",
        ],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)

    skills.to_csv(OUTPUT_FILE, index=False)

    print("\nO*NET target-occupation skill dataset created.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Total rows: {len(skills):,}")
    print(f"Unique skill elements: {skills['element_id'].nunique():,}")

    print("\nRows by target occupation and skill category:")
    print(
        skills.groupby(
            ["target_occupation", "onet_skill_category"]
        ).size().to_string()
    )

    print("\nMissing values in O*NET measures:")
    print(
        skills[["onet_importance", "onet_level"]]
        .isna()
        .sum()
        .to_string()
    )

    print("\nSample records:")
    print(skills.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
