import pandas as pd
from pathlib import Path

PROFILE_FILE = Path(
    "data/processed/user_skill_profile.csv"
)

MAPPING_FILE = Path(
    "data/processed/user_skill_esco_mapping.csv"
)

REPRESENTATION_FILE = Path(
    "data/processed/user_skill_representation.csv"
)

profile_df = pd.read_csv(PROFILE_FILE)
mapping_df = pd.read_csv(MAPPING_FILE)
representation_df = pd.read_csv(REPRESENTATION_FILE)

print("=" * 70)
print("PHASE 6.3 — VALIDATE USER SKILL REPRESENTATION")
print("=" * 70)

print(f"\nUser profile shape: {profile_df.shape}")
print(f"Canonical mapping shape: {mapping_df.shape}")
print(f"Representation shape: {representation_df.shape}")

required_columns = [
    "user_id",
    "user_skill",
    "esco_skill_uri",
    "esco_skill_label",
    "esco_skill_type",
    "esco_reuse_level",
    "skill_possession",
    "evidence_source",
]

for column in required_columns:
    if column not in representation_df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )

print("\n[PASS] All required columns are present.")

if len(representation_df) != len(profile_df):
    raise ValueError(
        "Representation row count does not match user profile."
    )

print("[PASS] Representation row count matches user profile.")

if representation_df["user_skill"].duplicated().any():
    raise ValueError(
        "Duplicate user skills found."
    )

print("[PASS] No duplicate user skills.")

if representation_df["esco_skill_uri"].isna().any():
    raise ValueError(
        "Missing ESCO skill URI found."
    )

print("[PASS] No missing ESCO skill URIs.")

if representation_df["esco_skill_uri"].nunique() != len(
    representation_df
):
    raise ValueError(
        "Multiple user skills map to the same ESCO skill."
    )

print("[PASS] Each user skill has a unique ESCO representation.")

if not representation_df["skill_possession"].isin([0, 1]).all():
    raise ValueError(
        "Skill possession contains values other than 0 or 1."
    )

print("[PASS] Skill possession is binary.")

if not (
    representation_df["skill_possession"] == 1
).all():
    raise ValueError(
        "Current user representation should contain "
        "only declared skills with possession=1."
    )

print("[PASS] All declared user skills have possession=1.")

if not (
    representation_df["evidence_source"]
    == "user_declared_skill"
).all():
    raise ValueError(
        "Unexpected evidence source found."
    )

print("[PASS] Evidence source is correctly recorded.")

profile_skills = set(profile_df["user_skill"])
representation_skills = set(
    representation_df["user_skill"]
)

if profile_skills != representation_skills:
    raise ValueError(
        "User skills in representation do not exactly "
        "match the user skill profile."
    )

print("[PASS] Representation exactly matches user skill profile.")

mapping_uris = set(mapping_df["esco_skill_uri"])
representation_uris = set(
    representation_df["esco_skill_uri"]
)

if mapping_uris != representation_uris:
    raise ValueError(
        "Representation ESCO URIs do not match canonical mapping."
    )

print("[PASS] Representation uses the canonical ESCO mappings.")

print("\nValidated user skill representation:")

for _, row in representation_df.iterrows():
    print(
        f"  {row['user_skill']}"
        f" -> "
        f"{row['esco_skill_label']}"
        f" | possession={row['skill_possession']}"
        f" | evidence={row['evidence_source']}"
    )

print("\nPHASE 6.3 VALIDATION: ALL CHECKS PASSED")