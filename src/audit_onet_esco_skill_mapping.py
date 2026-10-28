
from pathlib import Path
from difflib import SequenceMatcher
import re

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

ONET_FILE = PROCESSED / "onet_target_occupation_skills.csv"
ESCO_FILE = PROCESSED / "esco_occupation_skill_relations.csv"
OUTPUT_FILE = PROCESSED / "onet_esco_skill_mapping_candidates.csv"

TOP_N = 5
MIN_SIMILARITY_PERCENT = 35.0


def normalize(text):
    """Normalize labels for candidate generation only."""
    if pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_overlap(a, b):
    """Jaccard overlap of label tokens; not a semantic similarity score."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def main():
    if not ONET_FILE.exists():
        raise FileNotFoundError(f"Missing input: {ONET_FILE}")
    if not ESCO_FILE.exists():
        raise FileNotFoundError(f"Missing input: {ESCO_FILE}")

    onet = pd.read_csv(ONET_FILE)
    esco = pd.read_csv(ESCO_FILE)

    required_onet = {
        "element_id",
        "element_name",
        "onet_skill_category",
    }
    required_esco = {"skill_uri", "skill_label"}

    if missing := required_onet - set(onet.columns):
        raise ValueError(f"O*NET file missing columns: {sorted(missing)}")
    if missing := required_esco - set(esco.columns):
        raise ValueError(f"ESCO file missing columns: {sorted(missing)}")

    # One record per distinct O*NET skill element, not per occupation.
    onet_skills = (
        onet[
            ["element_id", "element_name", "onet_skill_category"]
        ]
        .drop_duplicates()
        .copy()
    )

    # The ESCO relations file provides the distinct skill labels used in
    # occupation-skill relationships; this is a candidate pool, not a
    # claim that it contains every possible ESCO skill.
    esco_skills = (
        esco[["skill_uri", "skill_label"]]
        .dropna(subset=["skill_uri", "skill_label"])
        .drop_duplicates(subset=["skill_uri"])
        .copy()
    )

    onet_skills["normalized_onet_label"] = (
        onet_skills["element_name"].map(normalize)
    )
    esco_skills["normalized_esco_label"] = (
        esco_skills["skill_label"].map(normalize)
    )

    esco_records = esco_skills.to_dict("records")
    output_rows = []

    for _, onet_row in onet_skills.iterrows():
        onet_label = onet_row["normalized_onet_label"]
        scored = []

        for esco_row in esco_records:
            esco_label = esco_row["normalized_esco_label"]

            sequence_score = SequenceMatcher(
                None, onet_label, esco_label
            ).ratio() * 100

            overlap_score = token_overlap(onet_label, esco_label) * 100
            exact = bool(onet_label and onet_label == esco_label)

            # Keep the metrics separate. Use the higher lexical score
            # only to rank candidates, not to declare equivalence.
            ranking_score = max(sequence_score, overlap_score)

            scored.append({
                "esco_skill_uri": esco_row["skill_uri"],
                "esco_skill_label": esco_row["skill_label"],
                "normalized_esco_label": esco_label,
                "string_similarity_percent": round(sequence_score, 2),
                "token_overlap_percent": round(overlap_score, 2),
                "exact_normalized_match": exact,
                "candidate_ranking_score": round(ranking_score, 2),
            })

        scored.sort(
            key=lambda x: (
                x["exact_normalized_match"],
                x["candidate_ranking_score"],
                x["token_overlap_percent"],
            ),
            reverse=True,
        )

        kept = [
            candidate for candidate in scored
            if candidate["exact_normalized_match"]
            or candidate["candidate_ranking_score"] >= MIN_SIMILARITY_PERCENT
        ][:TOP_N]

        if not kept:
            kept = scored[:1]

        for rank, candidate in enumerate(kept, start=1):
            output_rows.append({
                "onet_element_id": onet_row["element_id"],
                "onet_skill_label": onet_row["element_name"],
                "onet_skill_category": onet_row["onet_skill_category"],
                "candidate_rank": rank,
                **candidate,
                "mapping_status": (
                    "UNREVIEWED_EXACT"
                    if candidate["exact_normalized_match"]
                    else "UNREVIEWED_CANDIDATE"
                ),
                "review_reason": "",
            })

    result = pd.DataFrame(output_rows)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nO*NET–ESCO skill mapping candidate audit complete.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Unique O*NET skill elements: {len(onet_skills)}")
    print(f"Distinct ESCO skills searched: {len(esco_skills)}")
    print(f"Candidate rows: {len(result)}")
    print(
        "O*NET skills with exact label matches:",
        result.loc[
            result["exact_normalized_match"], "onet_element_id"
        ].nunique(),
    )
    print(
        "O*NET skills with no exact candidate:",
        result.loc[
            ~result["exact_normalized_match"], "onet_element_id"
        ].nunique(),
    )

    print("\nTop candidates per O*NET skill:")
    print(
        result[result["candidate_rank"] <= 3][[
            "onet_skill_label",
            "onet_skill_category",
            "candidate_rank",
            "esco_skill_label",
            "string_similarity_percent",
            "token_overlap_percent",
            "exact_normalized_match",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
