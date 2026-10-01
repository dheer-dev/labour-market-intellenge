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

MAPPING_FILE = Path(
    "data/processed/user_skill_esco_mapping.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

user_df = pd.read_csv(USER_FILE)
esco_df = pd.read_csv(ESCO_FILE)
mapping_df = pd.read_csv(MAPPING_FILE)


print("=" * 70)
print("PHASE 6.2.4 — VALIDATE USER / ESCO SKILL MAPPING")
print("=" * 70)

print(f"\nUser profile shape: {user_df.shape}")
print(f"ESCO skills shape: {esco_df.shape}")
print(f"Mapping shape: {mapping_df.shape}")


# ============================================================
# EXPECTED USER SKILLS
# ============================================================

expected_skills = set(
    user_df["user_skill"]
)

mapped_skills = set(
    mapping_df["user_skill"]
)


# ============================================================
# CHECK 1 — ROW COUNT
# ============================================================

if len(mapping_df) == len(user_df):
    print("\n[PASS] Mapping row count matches user profile.")
else:
    print("\n[FAIL] Mapping row count does not match user profile.")


# ============================================================
# CHECK 2 — NO DUPLICATE USER SKILLS
# ============================================================

duplicate_user_skills = mapping_df[
    mapping_df.duplicated(
        subset=["user_id", "user_skill"],
        keep=False
    )
]

if duplicate_user_skills.empty:
    print("[PASS] No duplicate user-skill mappings.")
else:
    print(
        "[FAIL] Duplicate user-skill mappings found:"
    )
    print(
        duplicate_user_skills[
            ["user_id", "user_skill"]
        ]
    )


# ============================================================
# CHECK 3 — ALL USER SKILLS MAPPED
# ============================================================

missing_skills = expected_skills - mapped_skills

if not missing_skills:
    print("[PASS] All user skills have mappings.")
else:
    print(
        f"[FAIL] Missing mappings: {missing_skills}"
    )


# ============================================================
# CHECK 4 — NO EXTRA SKILLS
# ============================================================

extra_skills = mapped_skills - expected_skills

if not extra_skills:
    print("[PASS] No unexpected user skills in mapping.")
else:
    print(
        f"[FAIL] Unexpected mapped skills: {extra_skills}"
    )


# ============================================================
# CHECK 5 — UNIQUE ESCO CONCEPTS
# ============================================================

unique_user_esco_pairs = mapping_df[
    ["user_skill", "esco_skill_uri"]
].drop_duplicates()

if (
    len(unique_user_esco_pairs)
    == len(mapping_df)
):
    print(
        "[PASS] Each user skill maps to one unique "
        "ESCO concept."
    )
else:
    print(
        "[FAIL] Duplicate user-skill / ESCO mappings found."
    )


# ============================================================
# CHECK 6 — ESCO URIS EXIST
# ============================================================

valid_esco_uris = set(
    esco_df["skill_uri"]
)

mapped_esco_uris = set(
    mapping_df["esco_skill_uri"]
)

invalid_uris = mapped_esco_uris - valid_esco_uris

if not invalid_uris:
    print("[PASS] All mapped ESCO URIs exist in ESCO skills.")
else:
    print(
        f"[FAIL] Invalid ESCO URIs: {invalid_uris}"
    )


# ============================================================
# CHECK 7 — LABELS MATCH ESCO
# ============================================================

esco_lookup = (
    esco_df[
        ["skill_uri", "skill_label"]
    ]
    .drop_duplicates(
        subset=["skill_uri"]
    )
    .set_index("skill_uri")["skill_label"]
    .to_dict()
)

label_mismatches = []

for _, row in mapping_df.iterrows():

    uri = row["esco_skill_uri"]
    mapped_label = row["esco_skill_label"]

    if uri in esco_lookup:

        if mapped_label != esco_lookup[uri]:

            label_mismatches.append(
                {
                    "user_skill": row["user_skill"],
                    "esco_skill_uri": uri,
                    "mapped_label": mapped_label,
                    "actual_label": esco_lookup[uri],
                }
            )


if not label_mismatches:
    print("[PASS] ESCO labels match their canonical URIs.")
else:
    print("[FAIL] ESCO label / URI mismatches found.")
    print(pd.DataFrame(label_mismatches))


# ============================================================
# CHECK 8 — REQUIRED FIELDS
# ============================================================

required_columns = [
    "user_id",
    "user_skill",
    "esco_skill_uri",
    "esco_skill_label",
    "esco_skill_type",
    "esco_reuse_level",
    "mapping_basis",
]

missing_columns = [
    column
    for column in required_columns
    if column not in mapping_df.columns
]

if not missing_columns:
    print("[PASS] All required mapping fields are present.")
else:
    print(
        f"[FAIL] Missing columns: {missing_columns}"
    )


# ============================================================
# FINAL MAPPING
# ============================================================

print("\nCanonical mapping:")

for _, row in mapping_df.iterrows():

    print(
        f"  {row['user_skill']}"
        f" -> "
        f"{row['esco_skill_label']}"
    )


# ============================================================
# FINAL STATUS
# ============================================================

checks = [
    len(mapping_df) == len(user_df),
    duplicate_user_skills.empty,
    not missing_skills,
    not extra_skills,
    (
        len(unique_user_esco_pairs)
        == len(mapping_df)
    ),
    not invalid_uris,
    not label_mismatches,
    not missing_columns,
]

print("\n" + "=" * 70)

if all(checks):
    print("PHASE 6.2.4 VALIDATION: ALL CHECKS PASSED")
else:
    print("PHASE 6.2.4 VALIDATION: CHECKS FAILED")

print("=" * 70)