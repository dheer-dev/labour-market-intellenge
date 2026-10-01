import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_FILE = Path(
    "data/processed/user_skill_profile.csv"
)


# ============================================================
# RESEARCH TEST USER
# ============================================================

user_id = "user_001"

user_skills = [
    "Python",
    "SQL",
    "Excel",
    "Communication",
    "Data Analysis",
]


# ============================================================
# BUILD USER SKILL DATASET
# ============================================================

records = []

for skill in user_skills:

    records.append(
        {
            "user_id": user_id,
            "user_skill": skill,
            "skill_source": "user_input",
        }
    )


df = pd.DataFrame(records)


# ============================================================
# VALIDATION
# ============================================================

if df.empty:
    raise ValueError(
        "User skill profile is empty."
    )

if df["user_skill"].isna().any():
    raise ValueError(
        "User skill profile contains missing skills."
    )

duplicate_skills = df.duplicated(
    subset=["user_id", "user_skill"]
).sum()

if duplicate_skills > 0:
    raise ValueError(
        f"Duplicate user skills found: "
        f"{duplicate_skills}"
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("=" * 70)
print("PHASE 6.1.1 — BUILD USER SKILL PROFILE")
print("=" * 70)

print(f"\nUser ID: {user_id}")
print(f"Number of user skills: {len(df)}")

print("\nUser skills:")

for skill in df["user_skill"]:
    print(f"  - {skill}")

print(f"\nOutput: {OUTPUT_FILE}")

print("\nPHASE 6.1.1: USER SKILL PROFILE BUILD COMPLETE")