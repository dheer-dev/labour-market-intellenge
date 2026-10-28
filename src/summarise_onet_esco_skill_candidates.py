
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    ROOT / "data" / "processed"
    / "onet_esco_skill_mapping_review_template.csv"
)
OUTPUT_FILE = (
    ROOT / "data" / "processed"
    / "onet_esco_skill_review_summary.csv"
)

df = pd.read_csv(INPUT_FILE)

candidate_columns = [
    "onet_element_id",
    "onet_skill_label",
    "onet_skill_category",
    "candidate_1_esco_label",
    "candidate_1_string_similarity",
    "candidate_1_token_overlap",
    "candidate_1_exact_match",
    "candidate_2_esco_label",
    "candidate_2_string_similarity",
    "candidate_2_token_overlap",
    "candidate_3_esco_label",
    "candidate_3_string_similarity",
    "candidate_3_token_overlap",
    "mapping_decision",
]

missing = [col for col in candidate_columns if col not in df.columns]
if missing:
    raise ValueError(
        f"Missing expected columns: {missing}\n"
        f"Actual columns: {df.columns.tolist()}"
    )

summary = df[candidate_columns].copy()
summary.to_csv(OUTPUT_FILE, index=False)

print(f"Saved: {OUTPUT_FILE}")
print(f"Rows: {len(summary)}")
print("\nO*NET skills and their top ESCO candidates:\n")

for _, row in summary.iterrows():
    print(f"\nO*NET: {row['onet_skill_label']}")
    print(f"Category: {row['onet_skill_category']}")

    for rank in range(1, 4):
        label = row.get(f"candidate_{rank}_esco_label")
        similarity = row.get(f"candidate_{rank}_string_similarity")
        overlap = row.get(f"candidate_{rank}_token_overlap")

        if pd.notna(label):
            print(
                f"  Candidate {rank}: {label} "
                f"| string similarity={similarity} "
                f"| token overlap={overlap}"
            )
