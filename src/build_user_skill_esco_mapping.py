import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

USER_FILE = Path(
    "data/processed/user_skill_profile.csv"
)

ESCO_FILE = Path(
    "data/processed/esco_skills.csv"
)

OUTPUT_FILE = Path(
    "data/processed/user_skill_esco_mapping.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

user_df = pd.read_csv(USER_FILE)
esco_df = pd.read_csv(ESCO_FILE)


print("=" * 70)
print("PHASE 6.2.3 — BUILD CANONICAL USER / ESCO SKILL MAPPING")
print("=" * 70)

print(f"\nUser profile shape: {user_df.shape}")
print(f"ESCO skills shape: {esco_df.shape}")


# ============================================================
# CANONICAL MAPPINGS
# ============================================================

mappings = {
    "Python": "ccd0a1d9-afda-43d9-b901-96344886e14d",
    "SQL": "598de5b0-5b58-4ea7-8058-a4bc4d18c742",
    "Excel": "1973c966-f236-40c9-b2d4-5d71a89019be",
    "Communication": "15d76317-c71a-4fa2-aadc-2ecc34e627b7",
    "Data Analysis": "2b92a5b2-6758-4ee3-9fb4-b6387a55cc8f",
}


# ============================================================
# VALIDATE USER SKILLS
# ============================================================

user_skills = set(
    user_df["user_skill"]
)

mapping_skills = set(
    mappings.keys()
)

missing_mappings = user_skills - mapping_skills
extra_mappings = mapping_skills - user_skills

if missing_mappings:
    raise ValueError(
        f"Missing mappings for user skills: "
        f"{missing_mappings}"
    )

if extra_mappings:
    raise ValueError(
        f"Mappings exist for skills not in user profile: "
        f"{extra_mappings}"
    )


# ============================================================
# BUILD MAPPING
# ============================================================

records = []

for _, user_row in user_df.iterrows():

    user_skill = user_row["user_skill"]

    esco_uri_suffix = mappings[user_skill]

    # Match canonical ESCO URI
    matches = esco_df[
        esco_df["skill_uri"].str.endswith(
            esco_uri_suffix
        )
    ]

    unique_matches = matches.drop_duplicates(
    subset=["skill_uri"]
)

    if len(unique_matches) != 1:
        raise ValueError(
            f"Expected exactly one unique ESCO skill URI for "
            f"{user_skill}, found {len(unique_matches)}"
        )

    esco_row = unique_matches.iloc[0]

    records.append(
        {
            "user_id": user_row["user_id"],
            "user_skill": user_skill,
            "esco_skill_uri": esco_row["skill_uri"],
            "esco_skill_label": esco_row["skill_label"],
            "esco_skill_type": esco_row["skill_type"],
            "esco_reuse_level": esco_row["reuse_level"],
            "mapping_basis": (
                "manual_semantic_mapping_from_ESCo_vocabulary"
            ),
        }
    )


# ============================================================
# CREATE OUTPUT
# ============================================================

mapping_df = pd.DataFrame(records)

mapping_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print(
    f"\nMapped user skills: "
    f"{len(mapping_df)}"
)

print("\nCanonical mappings:")

for _, row in mapping_df.iterrows():

    print(
        f"  {row['user_skill']}"
        f" -> "
        f"{row['esco_skill_label']}"
    )

print(
    f"\nUnique user skills: "
    f"{mapping_df['user_skill'].nunique()}"
)

print(
    f"Unique ESCO skills: "
    f"{mapping_df['esco_skill_uri'].nunique()}"
)

print(f"\nOutput: {OUTPUT_FILE}")

print(
    "\nPHASE 6.2.3: CANONICAL MAPPING COMPLETE"
)