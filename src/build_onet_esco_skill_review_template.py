
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

INPUT_FILE = PROCESSED / "onet_esco_skill_mapping_candidates.csv"
OUTPUT_FILE = PROCESSED / "onet_esco_skill_mapping_review_template.csv"

df = pd.read_csv(INPUT_FILE)

# Keep the three highest-ranked suggestions for each O*NET skill.
df = df.sort_values(
    ["onet_element_id", "candidate_rank"]
)

base = (
    df.drop_duplicates("onet_element_id")
    [["onet_element_id", "onet_skill_label", "onet_skill_category"]]
    .copy()
)

for rank in range(1, 4):
    candidates = df[df["candidate_rank"] == rank][[
        "onet_element_id",
        "esco_skill_uri",
        "esco_skill_label",
        "string_similarity_percent",
        "token_overlap_percent",
        "exact_normalized_match",
    ]].copy()

    candidates = candidates.rename(columns={
        "esco_skill_uri": f"candidate_{rank}_esco_uri",
        "esco_skill_label": f"candidate_{rank}_esco_label",
        "string_similarity_percent":
            f"candidate_{rank}_string_similarity",
        "token_overlap_percent":
            f"candidate_{rank}_token_overlap",
        "exact_normalized_match":
            f"candidate_{rank}_exact_match",
    })

    base = base.merge(
        candidates,
        on="onet_element_id",
        how="left",
        validate="one_to_one",
    )

# These fields must be completed through conceptual review.
base["reviewed_esco_skill_uri"] = ""
base["reviewed_esco_skill_label"] = ""
base["mapping_decision"] = "UNREVIEWED"
base["mapping_rationale"] = ""

base.to_csv(OUTPUT_FILE, index=False)

skills_with_exact = set(
    df.loc[df["exact_normalized_match"], "onet_element_id"]
)
all_skills = set(df["onet_element_id"])

print(f"Saved review template: {OUTPUT_FILE}")
print(f"Unique O*NET skills: {len(all_skills)}")
print(f"Skills with exact label matches: {len(skills_with_exact)}")
print(f"Skills without exact label matches: "
      f"{len(all_skills - skills_with_exact)}")
print(f"Review rows: {len(base)}")
print("\nDecision options:")
print("APPROVED, REJECTED, NO_DIRECT_MATCH, REVIEW_REQUIRED")
