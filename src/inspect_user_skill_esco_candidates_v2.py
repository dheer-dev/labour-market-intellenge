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


# ============================================================
# LOAD DATA
# ============================================================

user_df = pd.read_csv(USER_FILE)
esco_df = pd.read_csv(ESCO_FILE)

print("=" * 70)
print("PHASE 6.2.2 — SEMANTIC INSPECTION OF ESCO SKILL CANDIDATES")
print("=" * 70)

print(f"\nUser profile shape: {user_df.shape}")
print(f"ESCO skills shape: {esco_df.shape}")


# ============================================================
# NORMALISATION
# ============================================================

def normalise_text(value):
    return (
        str(value)
        .lower()
        .strip()
        .replace("-", " ")
        .replace("/", " ")
    )


esco_df["label_normalised"] = (
    esco_df["skill_label"]
    .apply(normalise_text)
)

esco_df["aliases_normalised"] = (
    esco_df["skill_aliases"]
    .fillna("")
    .apply(normalise_text)
)


# ============================================================
# INSPECT EACH USER SKILL
# ============================================================

for user_skill in user_df["user_skill"]:

    skill_norm = normalise_text(user_skill)

    print("\n" + "-" * 70)
    print(f"USER SKILL: {user_skill}")
    print("-" * 70)

    # --------------------------------------------------------
    # Exact label matches
    # --------------------------------------------------------

    exact_matches = esco_df[
        esco_df["label_normalised"] == skill_norm
    ]

    print(
        f"\nEXACT LABEL MATCHES: "
        f"{len(exact_matches)}"
    )

    for _, row in exact_matches.iterrows():

        print(
            f"\n  ESCO label: {row['skill_label']}"
            f"\n  URI: {row['skill_uri']}"
            f"\n  Type: {row['skill_type']}"
            f"\n  Reuse level: {row['reuse_level']}"
            f"\n  Aliases: {row['skill_aliases']}"
            f"\n  Description: {row['skill_description']}"
        )

    # --------------------------------------------------------
    # Alias matches
    # --------------------------------------------------------

    alias_matches = esco_df[
        esco_df["aliases_normalised"].apply(
            lambda aliases:
                skill_norm in aliases
                if aliases
                else False
        )
    ]

    # Remove exact-label matches
    alias_matches = alias_matches[
        ~alias_matches["skill_uri"].isin(
            exact_matches["skill_uri"]
        )
    ]

    print(
        f"\nALIAS CONTAINS MATCHES: "
        f"{len(alias_matches)}"
    )

    for _, row in alias_matches.head(20).iterrows():

        print(
            f"\n  ESCO label: {row['skill_label']}"
            f"\n  URI: {row['skill_uri']}"
            f"\n  Type: {row['skill_type']}"
            f"\n  Reuse level: {row['reuse_level']}"
            f"\n  Aliases: {row['skill_aliases']}"
            f"\n  Description: {row['skill_description']}"
        )


print("\n" + "=" * 70)
print("PHASE 6.2.2: SEMANTIC CANDIDATE INSPECTION COMPLETE")
print("=" * 70)