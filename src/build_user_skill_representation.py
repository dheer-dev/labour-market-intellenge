import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

MAPPING_FILE = Path(
    "data/processed/user_skill_esco_mapping.csv"
)

OUTPUT_FILE = Path(
    "data/processed/user_skill_representation.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

mapping_df = pd.read_csv(MAPPING_FILE)

print("=" * 70)
print("PHASE 6.3 — BUILD USER SKILL REPRESENTATION")
print("=" * 70)

print(
    f"\nCanonical mapping shape: "
    f"{mapping_df.shape}"
)


# ============================================================
# BUILD USER SKILL REPRESENTATION
# ============================================================

representation_df = mapping_df[
    [
        "user_id",
        "user_skill",
        "esco_skill_uri",
        "esco_skill_label",
        "esco_skill_type",
        "esco_reuse_level",
    ]
].copy()


# Binary possession indicator.
# The current research profile explicitly states that
# each listed skill is possessed by the user.

representation_df["skill_possession"] = 1


# Evidence source records how the representation was created.
representation_df["evidence_source"] = (
    "user_declared_skill"
)


# ============================================================
# VALIDATION
# ============================================================

if representation_df.empty:
    raise ValueError(
        "User skill representation is empty."
    )

if representation_df["esco_skill_uri"].isna().any():
    raise ValueError(
        "Missing ESCO skill URI found."
    )

if representation_df["skill_possession"].isin(
    [0, 1]
).all() is False:
    raise ValueError(
        "Skill possession must be binary."
    )


# ============================================================
# SAVE
# ============================================================

representation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nNumber of represented user skills: "
    f"{len(representation_df)}"
)

print(
    f"Unique ESCO skills represented: "
    f"{representation_df['esco_skill_uri'].nunique()}"
)

print("\nUser skill representation:")

for _, row in representation_df.iterrows():

    print(
        f"  {row['user_skill']}"
        f" -> "
        f"{row['esco_skill_label']}"
        f" | possession={row['skill_possession']}"
    )

print(f"\nOutput: {OUTPUT_FILE}")

print(
    "\nPHASE 6.3: USER SKILL REPRESENTATION BUILD COMPLETE"
)