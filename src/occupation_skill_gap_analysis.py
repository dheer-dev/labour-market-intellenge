from pathlib import Path
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MARKET_FILE = PROCESSED_DIR / "occupation_skill_demand_analysis.csv"
ESCO_FILE = PROCESSED_DIR / "esco_occupation_skill_relations.csv"

OUTPUT_FILE = PROCESSED_DIR / "occupation_skill_gap_analysis.csv"
SUMMARY_FILE = PROCESSED_DIR / "occupation_skill_gap_summary.csv"


SELECTED_OCCUPATIONS = [
    "project manager",
    "civil engineer",
    "lawyer",
    "environmental engineer",
    "construction manager",
]


# Market demand threshold
# Skills at or above this prevalence are treated as
# "market-supported" for the gap summary.
MARKET_DEMAND_THRESHOLD = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHASE 4 — SKILL GAP ANALYSIS")
print("=" * 70)

market = pd.read_csv(MARKET_FILE)
esco = pd.read_csv(ESCO_FILE)

print(f"\nMarket skill-demand shape: {market.shape}")
print(f"ESCO occupation-skill relations shape: {esco.shape}")


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_market_cols = [
    "occupation_uri",
    "occupation_label",
    "skill_uri",
    "skill_label",
    "skill_posting_count",
    "postings_with_skills",
    "skill_prevalence",
    "skill_prevalence_percent",
    "skill_rank",
    "demand_category",
]

required_esco_cols = [
    "occupation_uri",
    "occupation_label",
    "relation_type",
    "skill_type",
    "skill_uri",
    "skill_label",
]


missing_market = [
    col for col in required_market_cols
    if col not in market.columns
]

missing_esco = [
    col for col in required_esco_cols
    if col not in esco.columns
]

if missing_market:
    raise ValueError(
        f"Missing columns in market data: {missing_market}"
    )

if missing_esco:
    raise ValueError(
        f"Missing columns in ESCO data: {missing_esco}"
    )


# ============================================================
# NORMALIZE LABELS
# ============================================================

market["occupation_label_norm"] = (
    market["occupation_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

esco["occupation_label_norm"] = (
    esco["occupation_label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


selected_set = set(SELECTED_OCCUPATIONS)


# ============================================================
# FILTER SELECTED OCCUPATIONS
# ============================================================

market = market[
    market["occupation_label_norm"].isin(selected_set)
].copy()

esco = esco[
    esco["occupation_label_norm"].isin(selected_set)
].copy()


print(
    f"\nMarket occupations: "
    f"{market['occupation_label'].nunique()}"
)

print(
    f"ESCO occupations: "
    f"{esco['occupation_label'].nunique()}"
)


# ============================================================
# NORMALIZE URIS
# ============================================================

for df in [market, esco]:

    df["occupation_uri"] = (
        df["occupation_uri"]
        .astype(str)
        .str.strip()
    )

    df["skill_uri"] = (
        df["skill_uri"]
        .astype(str)
        .str.strip()
    )


# ============================================================
# VALIDATE ESCO RELATIONS
# ============================================================

esco_duplicates = esco.duplicated(
    subset=[
        "occupation_uri",
        "skill_uri",
        "relation_type",
        "skill_type",
    ]
).sum()

print(
    f"\nDuplicate ESCO occupation-skill relations: "
    f"{esco_duplicates}"
)

if esco_duplicates > 0:
    raise ValueError(
        "Duplicate ESCO occupation-skill relations found."
    )


# ============================================================
# BUILD ESCO SKILL SET PER OCCUPATION
# ============================================================

esco_lookup = {}

for occupation_uri, group in esco.groupby(
    "occupation_uri"
):

    esco_lookup[occupation_uri] = set(
        group["skill_uri"]
    )


# ============================================================
# BUILD MARKET SKILL SET PER OCCUPATION
# ============================================================

market_lookup = {}

for occupation_uri, group in market.groupby(
    "occupation_uri"
):

    market_lookup[occupation_uri] = set(
        group["skill_uri"]
    )


# ============================================================
# GAP CLASSIFICATION
# ============================================================

gap_rows = []


for occupation_uri in sorted(
    set(market_lookup.keys()) | set(esco_lookup.keys())
):

    market_group = market[
        market["occupation_uri"] == occupation_uri
    ].copy()

    esco_group = esco[
        esco["occupation_uri"] == occupation_uri
    ].copy()

    occupation_label = (
        market_group["occupation_label"].iloc[0]
        if len(market_group) > 0
        else esco_group["occupation_label"].iloc[0]
    )

    market_skills = market_lookup.get(
        occupation_uri,
        set()
    )

    esco_skills = esco_lookup.get(
        occupation_uri,
        set()
    )

    # --------------------------------------------------------
    # OVERLAP
    # --------------------------------------------------------

    overlap_skills = market_skills & esco_skills

    # --------------------------------------------------------
    # MARKET-ONLY
    # --------------------------------------------------------

    market_only_skills = market_skills - esco_skills

    # --------------------------------------------------------
    # ESCO-ONLY
    # --------------------------------------------------------

    esco_only_skills = esco_skills - market_skills


    # --------------------------------------------------------
    # MARKET + ESCO OVERLAP
    # --------------------------------------------------------

    for skill_uri in overlap_skills:

        market_row = market_group[
            market_group["skill_uri"] == skill_uri
        ].iloc[0]

        esco_rows = esco_group[
            esco_group["skill_uri"] == skill_uri
        ]

        for _, esco_row in esco_rows.iterrows():

            gap_rows.append({

                "occupation_uri":
                    occupation_uri,

                "occupation_label":
                    occupation_label,

                "skill_uri":
                    skill_uri,

                "skill_label":
                    market_row["skill_label"],

                "market_skill_posting_count":
                    market_row["skill_posting_count"],

                "market_skill_prevalence":
                    market_row["skill_prevalence"],

                "market_skill_prevalence_percent":
                    market_row["skill_prevalence_percent"],

                "market_skill_rank":
                    market_row["skill_rank"],

                "market_demand_category":
                    market_row["demand_category"],

                "esco_relation_type":
                    esco_row["relation_type"],

                "esco_skill_type":
                    esco_row["skill_type"],

                "gap_category":
                    "Market + ESCO",

                "market_supported":
                    market_row["skill_prevalence"]
                    >= MARKET_DEMAND_THRESHOLD,

            })


    # --------------------------------------------------------
    # MARKET-ONLY SKILLS
    # --------------------------------------------------------

    for skill_uri in market_only_skills:

        market_row = market_group[
            market_group["skill_uri"] == skill_uri
        ].iloc[0]

        gap_rows.append({

            "occupation_uri":
                occupation_uri,

            "occupation_label":
                occupation_label,

            "skill_uri":
                skill_uri,

            "skill_label":
                market_row["skill_label"],

            "market_skill_posting_count":
                market_row["skill_posting_count"],

            "market_skill_prevalence":
                market_row["skill_prevalence"],

            "market_skill_prevalence_percent":
                market_row["skill_prevalence_percent"],

            "market_skill_rank":
                market_row["skill_rank"],

            "market_demand_category":
                market_row["demand_category"],

            "esco_relation_type":
                None,

            "esco_skill_type":
                None,

            "gap_category":
                "Market-only",

            "market_supported":
                market_row["skill_prevalence"]
                >= MARKET_DEMAND_THRESHOLD,

        })


    # --------------------------------------------------------
    # ESCO-ONLY SKILLS
    # --------------------------------------------------------

    for skill_uri in esco_only_skills:

        esco_rows = esco_group[
            esco_group["skill_uri"] == skill_uri
        ]

        for _, esco_row in esco_rows.iterrows():

            gap_rows.append({

                "occupation_uri":
                    occupation_uri,

                "occupation_label":
                    occupation_label,

                "skill_uri":
                    skill_uri,

                "skill_label":
                    esco_row["skill_label"],

                "market_skill_posting_count":
                    0,

                "market_skill_prevalence":
                    0.0,

                "market_skill_prevalence_percent":
                    0.0,

                "market_skill_rank":
                    None,

                "market_demand_category":
                    None,

                "esco_relation_type":
                    esco_row["relation_type"],

                "esco_skill_type":
                    esco_row["skill_type"],

                "gap_category":
                    "ESCO-only",

                "market_supported":
                    False,

            })


# ============================================================
# CREATE OUTPUT
# ============================================================

gap_analysis = pd.DataFrame(gap_rows)


gap_analysis = gap_analysis.sort_values(
    by=[
        "occupation_label",
        "gap_category",
        "market_skill_prevalence",
        "skill_label",
    ],
    ascending=[
        True,
        True,
        False,
        True,
    ]
).reset_index(drop=True)


gap_analysis.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []


for occupation_uri, group in gap_analysis.groupby(
    "occupation_uri",
    sort=False
):

    occupation_label = (
        group["occupation_label"].iloc[0]
    )

    overlap = group[
        group["gap_category"] == "Market + ESCO"
    ]

    market_only = group[
        group["gap_category"] == "Market-only"
    ]

    esco_only = group[
        group["gap_category"] == "ESCO-only"
    ]

    market_supported = group[
        group["market_supported"] == True
    ]

    market_supported_market_only = market_only[
        market_only["market_supported"] == True
    ]

    top_market_only = (
        market_supported_market_only
        .sort_values(
            "market_skill_prevalence",
            ascending=False
        )
        .head(10)
    )

    top_market_only_names = (
        top_market_only["skill_label"]
        .tolist()
    )

    summary_rows.append({

        "occupation_uri":
            occupation_uri,

        "occupation_label":
            occupation_label,

        "market_skill_count":
            len(market_skills := set(
                market[
                    market["occupation_uri"]
                    == occupation_uri
                ]["skill_uri"]
            )),

        "esco_skill_count":
            len(set(
                esco[
                    esco["occupation_uri"]
                    == occupation_uri
                ]["skill_uri"]
            )),

        "market_esco_overlap_count":
            len(overlap),

        "market_only_skill_count":
            len(market_only),

        "esco_only_skill_count":
            len(esco_only),

        "market_supported_skill_count":
            len(market_supported),

        "market_supported_market_only_count":
            len(market_supported_market_only),

        "market_supported_market_only_skills":
            " | ".join(top_market_only_names),

        "market_to_esco_coverage_percent":
            (
                len(overlap) / len(market_skills) * 100
                if len(market_skills) > 0
                else 0
            ),

        "esco_to_market_coverage_percent":
            (
                len(overlap) / len(set(
                    esco[
                        esco["occupation_uri"]
                        == occupation_uri
                    ]["skill_uri"]
                )) * 100
                if len(esco_skills := set(
                    esco[
                        esco["occupation_uri"]
                        == occupation_uri
                    ]["skill_uri"]
                )) > 0
                else 0
            ),
    })


summary = pd.DataFrame(summary_rows)


# Preserve selected occupation order

occupation_order = {
    occupation: i
    for i, occupation in enumerate(
        SELECTED_OCCUPATIONS
    )
}

summary["sort_order"] = (
    summary["occupation_label"]
    .str.lower()
    .map(occupation_order)
)

summary = (
    summary
    .sort_values("sort_order")
    .drop(columns=["sort_order"])
    .reset_index(drop=True)
)


summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SKILL GAP SUMMARY")
print("=" * 70)

print(
    summary[
        [
            "occupation_label",
            "market_skill_count",
            "esco_skill_count",
            "market_esco_overlap_count",
            "market_only_skill_count",
            "esco_only_skill_count",
            "market_supported_market_only_count",
            "market_to_esco_coverage_percent",
        ]
    ].to_string(index=False)
)


# ============================================================
# PRINT MARKET-SUPPORTED MARKET-ONLY SKILLS
# ============================================================

print("\n" + "=" * 70)
print("MARKET-SUPPORTED SKILLS NOT IN ESCO")
print("=" * 70)

for occupation in SELECTED_OCCUPATIONS:

    group = gap_analysis[
        (
            gap_analysis["occupation_label"]
            .str.lower() == occupation
        )
        &
        (
            gap_analysis["gap_category"]
            == "Market-only"
        )
        &
        (
            gap_analysis["market_supported"]
            == True
        )
    ].sort_values(
        "market_skill_prevalence",
        ascending=False
    ).head(10)

    print(f"\n{occupation.upper()}")

    if len(group) == 0:

        print(
            "No market-supported market-only skills."
        )

    else:

        print(
            group[
                [
                    "skill_label",
                    "market_skill_posting_count",
                    "market_skill_prevalence_percent",
                    "market_demand_category",
                ]
            ].to_string(index=False)
        )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(
    f"Occupations analysed: "
    f"{gap_analysis['occupation_label'].nunique()}"
)

print(
    f"Total occupation-skill rows: "
    f"{len(gap_analysis)}"
)

print(
    f"Gap categories: "
    f"{gap_analysis['gap_category'].unique().tolist()}"
)

print(
    f"\nDetailed output saved to:\n"
    f"{OUTPUT_FILE}"
)

print(
    f"\nSummary output saved to:\n"
    f"{SUMMARY_FILE}"
)

print("\nPhase 4 completed successfully.")