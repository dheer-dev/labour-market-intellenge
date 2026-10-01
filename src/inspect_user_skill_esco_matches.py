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
    "data/processed/user_skill_esco_match_candidates.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

user_df = pd.read_csv(USER_FILE)

esco_df = pd.read_csv(ESCO_FILE)

print("=" * 70)
print("PHASE 6.2.1 — INSPECT USER SKILL / ESCO MATCH CANDIDATES")
print("=" * 70)

print(f"\nUser profile shape: {user_df.shape}")
print(f"ESCO skills shape: {esco_df.shape}")


# ============================================================
# NORMALISE TEXT FOR SEARCH
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
# FIND CANDIDATES
# ============================================================

records = []

for _, user_row in user_df.iterrows():

    user_skill = user_row["user_skill"]
    user_skill_norm = normalise_text(user_skill)

    # --------------------------------------------------------
    # Exact label match
    # --------------------------------------------------------

    exact_label = esco_df[
        esco_df["label_normalised"]
        == user_skill_norm
    ].copy()

    for _, esco_row in exact_label.iterrows():

        records.append(
            {
                "user_id": user_row["user_id"],
                "user_skill": user_skill,
                "match_type": "exact_label",
                "esco_skill_uri": esco_row["skill_uri"],
                "esco_skill_label": esco_row["skill_label"],
                "esco_skill_aliases": esco_row["skill_aliases"],
                "esco_skill_type": esco_row["skill_type"],
                "esco_reuse_level": esco_row["reuse_level"],
            }
        )

    # --------------------------------------------------------
    # Alias match
    # --------------------------------------------------------

    alias_matches = esco_df[
        esco_df["aliases_normalised"].apply(
            lambda aliases:
                user_skill_norm in aliases.split()
        )
    ].copy()

    # The word-level check above is intentionally conservative.
    # We additionally check whether the complete user phrase
    # occurs inside the alias text.

    alias_matches = esco_df[
        esco_df["aliases_normalised"].apply(
            lambda aliases:
                user_skill_norm in aliases
                if aliases
                else False
        )
    ].copy()

    for _, esco_row in alias_matches.iterrows():

        # Avoid duplicating exact-label matches.
        if (
            esco_row["skill_uri"]
            in set(exact_label["skill_uri"])
        ):
            continue

        records.append(
            {
                "user_id": user_row["user_id"],
                "user_skill": user_skill,
                "match_type": "alias_contains",
                "esco_skill_uri": esco_row["skill_uri"],
                "esco_skill_label": esco_row["skill_label"],
                "esco_skill_aliases": esco_row["skill_aliases"],
                "esco_skill_type": esco_row["skill_type"],
                "esco_reuse_level": esco_row["reuse_level"],
            }
        )


# ============================================================
# CREATE OUTPUT
# ============================================================

matches_df = pd.DataFrame(records)

if matches_df.empty:
    print("\nNo exact-label or alias candidates found.")
else:

    matches_df = matches_df.drop_duplicates(
        subset=[
            "user_skill",
            "esco_skill_uri",
            "match_type",
        ]
    )

    matches_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


# ============================================================
# SUMMARY
# ============================================================

print(
    f"\nCandidate matches found: "
    f"{len(matches_df)}"
)

print("\nMatches by user skill:")

for skill in user_df["user_skill"]:

    skill_matches = matches_df[
        matches_df["user_skill"] == skill
    ]

    print(
        f"\n{skill}: "
        f"{len(skill_matches)} candidate(s)"
    )

    for _, row in skill_matches.head(10).iterrows():

        print(
            f"  [{row['match_type']}] "
            f"{row['esco_skill_label']}"
        )


# ============================================================
# OUTPUT
# ============================================================

print(f"\nOutput: {OUTPUT_FILE}")

print(
    "\nPHASE 6.2.1: MATCH INSPECTION COMPLETE"
)