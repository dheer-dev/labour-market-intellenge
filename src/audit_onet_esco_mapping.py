
from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ESCO_FILE = PROCESSED / "esco_occupation_skill_relations.csv"
ONET_OCC_FILE = PROCESSED / "onet_occupations_clean.csv"
ONET_TITLES_FILE = PROCESSED / "onet_job_titles_clean.csv"

OUTPUT_FILE = PROCESSED / "onet_esco_occupation_mapping_candidates.csv"

TARGETS = [
    "project manager",
    "civil engineer",
    "lawyer",
    "environmental engineer",
    "construction manager",
]


def normalize(value):
    """Normalize text for comparison, without asserting equivalence."""
    if pd.isna(value):
        return ""
    value = str(value).casefold().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def similarity(a, b):
    """Transparent string similarity; not a validated mapping probability."""
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    return round(100 * SequenceMatcher(None, a, b).ratio(), 2)


def main():
    for path in [ESCO_FILE, ONET_OCC_FILE, ONET_TITLES_FILE]:
        if not path.exists():
            raise FileNotFoundError(f"Required input not found: {path}")

    esco = pd.read_csv(ESCO_FILE)
    onet = pd.read_csv(ONET_OCC_FILE)
    titles = pd.read_csv(ONET_TITLES_FILE)

    required_esco = {"occupation_uri", "occupation_label"}
    required_onet = {"onet_soc_code", "title"}
    required_titles = {"onet_soc_code", "job_title"}

    if not required_esco.issubset(esco.columns):
        raise ValueError(f"Unexpected ESCO columns: {esco.columns.tolist()}")
    if not required_onet.issubset(onet.columns):
        raise ValueError(f"Unexpected O*NET columns: {onet.columns.tolist()}")
    if not required_titles.issubset(titles.columns):
        raise ValueError(f"Unexpected O*NET title columns: {titles.columns.tolist()}")

    esco_occ = esco[
        ["occupation_uri", "occupation_label"]
    ].drop_duplicates()

    onet_occ = onet[
        ["onet_soc_code", "title"]
    ].drop_duplicates()

    # Keep alternate job titles attached to their O*NET occupation code.
    title_rows = titles[
        ["onet_soc_code", "job_title"]
    ].dropna(subset=["job_title"]).drop_duplicates()

    aliases = (
        title_rows.groupby("onet_soc_code")["job_title"]
        .agg(list)
        .to_dict()
    )

    rows = []

    for target in TARGETS:
        matching_esco = esco_occ[
            esco_occ["occupation_label"].map(normalize) == normalize(target)
        ]

        if matching_esco.empty:
            print(f"WARNING: ESCO target label not found exactly: {target}")
            continue

        for _, esco_row in matching_esco.iterrows():
            esco_label = esco_row["occupation_label"]

            for _, onet_row in onet_occ.iterrows():
                soc = onet_row["onet_soc_code"]
                onet_title = onet_row["title"]
                alt_titles = aliases.get(soc, [])

                title_candidates = [onet_title] + alt_titles
                scored = [
                    (similarity(esco_label, title), title)
                    for title in title_candidates
                ]
                best_score, best_title = max(
                    scored, key=lambda item: item[0]
                )

                exact = (
                    normalize(esco_label) == normalize(best_title)
                )

                # Retain plausible candidates for inspection.
                if exact or best_score >= 65:
                    rows.append({
                        "target_occupation": target,
                        "esco_occupation_uri": esco_row["occupation_uri"],
                        "esco_occupation_label": esco_label,
                        "onet_soc_code": soc,
                        "onet_occupation_title": onet_title,
                        "best_matching_onet_title": best_title,
                        "string_similarity_percent": best_score,
                        "exact_normalized_match": exact,
                        "candidate_basis": (
                            "exact normalized title"
                            if exact
                            else "string similarity; manual verification required"
                        ),
                        "alternative_title_count": len(alt_titles),
                        "mapping_status": "UNREVIEWED",
                    })

    result = pd.DataFrame(rows)

    if result.empty:
        print("No candidate mappings generated; inspect labels and input files.")
        return

    result = result.sort_values(
        ["target_occupation", "string_similarity_percent"],
        ascending=[True, False],
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nCandidate mapping audit complete.")
    print("Output:", OUTPUT_FILE)
    print("Candidate rows:", len(result))
    print("\nCandidates by target:")
    print(
        result.groupby("target_occupation").size().to_string()
    )
    print("\nBest candidates per target:")
    print(
        result.groupby("target_occupation", group_keys=False)
        .head(5)[[
            "target_occupation",
            "esco_occupation_label",
            "onet_soc_code",
            "onet_occupation_title",
            "best_matching_onet_title",
            "string_similarity_percent",
            "exact_normalized_match",
        ]].to_string(index=False)
    )
    print(
        "\nIMPORTANT: similarity scores are candidate-generation aids, "
        "not validated occupational equivalence."
    )


if __name__ == "__main__":
    main()
