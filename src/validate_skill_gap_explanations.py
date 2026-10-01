import pandas as pd
import re
from pathlib import Path


# ==============================================================
# PHASE 6.2 — EXPLANATION VALIDATION
# ==============================================================

INPUT_PATH = Path(
    "data/processed/skill_gap_explanations.csv"
)


print("=" * 70)
print("PHASE 6.2 — SKILL-GAP EXPLANATION VALIDATION")
print("=" * 70)


df = pd.read_csv(INPUT_PATH)

print(f"\nInput shape: {df.shape}")


# --------------------------------------------------------------
# CHECK 1 — ROW COUNT
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("1. ROW COUNT VALIDATION")
print("=" * 70)

expected_rows = 35
actual_rows = len(df)

print(f"Expected: {expected_rows}")
print(f"Actual:   {actual_rows}")

row_count_pass = actual_rows == expected_rows

print("PASS" if row_count_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 2 — DUPLICATES
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("2. DUPLICATE VALIDATION")
print("=" * 70)

duplicates = df.duplicated(
    subset=["occupation_label", "skill_label"]
).sum()

print(f"Duplicate occupation-skill pairs: {duplicates}")

duplicates_pass = duplicates == 0

print("PASS" if duplicates_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 3 — GAP CATEGORY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("3. GAP CATEGORY VALIDATION")
print("=" * 70)

invalid_gap_categories = set(
    df["gap_category"].dropna().unique()
) - {"Market-only"}

print(f"Invalid gap categories: {invalid_gap_categories}")

gap_category_pass = len(invalid_gap_categories) == 0

print("PASS" if gap_category_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 4 — MARKET SUPPORTED
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("4. MARKET SUPPORT VALIDATION")
print("=" * 70)

invalid_supported = df[
    df["market_supported"] != True
]

print(f"Rows not marked market-supported: {len(invalid_supported)}")

market_supported_pass = len(invalid_supported) == 0

print("PASS" if market_supported_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 5 — PRIORITY LABELS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("5. PRIORITY LABEL VALIDATION")
print("=" * 70)

valid_priorities = {
    "High",
    "Moderate",
    "Emerging",
}

invalid_priorities = set(
    df["priority"].dropna().unique()
) - valid_priorities

print(f"Invalid priority labels: {invalid_priorities}")

priority_labels_pass = len(invalid_priorities) == 0

print("PASS" if priority_labels_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 6 — PRIORITY REPRODUCIBILITY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("6. PRIORITY RULE REPRODUCIBILITY")
print("=" * 70)


def expected_priority(prevalence):
    if prevalence >= 40:
        return "High"
    elif prevalence >= 30:
        return "Moderate"
    else:
        return "Emerging"


df["expected_priority"] = df[
    "market_skill_prevalence_percent"
].apply(expected_priority)

priority_mismatches = df[
    df["priority"] != df["expected_priority"]
]

print(f"Priority mismatches: {len(priority_mismatches)}")

priority_rule_pass = len(priority_mismatches) == 0

print("PASS" if priority_rule_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 7 — PREVALENCE / POSTING COUNT / RANK VALIDITY
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("7. EVIDENCE FIELD VALIDATION")
print("=" * 70)

invalid_prevalence = df[
    (df["market_skill_prevalence_percent"] < 0)
    | (df["market_skill_prevalence_percent"] > 100)
].shape[0]

missing_posting_count = df[
    df["market_skill_posting_count"].isna()
].shape[0]

missing_rank = df[
    df["market_skill_rank"].isna()
].shape[0]

print(f"Invalid prevalence values: {invalid_prevalence}")
print(f"Missing posting counts:    {missing_posting_count}")
print(f"Missing market ranks:      {missing_rank}")

evidence_fields_pass = (
    invalid_prevalence == 0
    and missing_posting_count == 0
    and missing_rank == 0
)

print("PASS" if evidence_fields_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 8 — EXPLANATION COMPLETENESS
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("8. EXPLANATION COMPLETENESS")
print("=" * 70)

missing_explanations = df[
    df["evidence_explanation"].isna()
].shape[0]

empty_explanations = df[
    df["evidence_explanation"]
    .fillna("")
    .str.strip()
    .eq("")
].shape[0]

print(f"Missing explanations: {missing_explanations}")
print(f"Empty explanations:   {empty_explanations}")

explanation_completeness_pass = (
    missing_explanations == 0
    and empty_explanations == 0
)

print(
    "PASS"
    if explanation_completeness_pass
    else "FAIL"
)


# --------------------------------------------------------------
# CHECK 9 — EXPLANATION CONTENT
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("9. EXPLANATION CONTENT VALIDATION")
print("=" * 70)


def explanation_contains_row_evidence(row):
    explanation = str(row["evidence_explanation"]).lower()

    occupation = str(
        row["occupation_label"]
    ).lower()

    skill = str(
        row["skill_label"]
    ).lower()

    priority = str(
        row["priority"]
    ).lower()

    return (
        occupation in explanation
        and skill in explanation
        and priority in explanation
        and "market-only" in explanation
        and "market postings" in explanation
    )


df["explanation_content_valid"] = df.apply(
    explanation_contains_row_evidence,
    axis=1
)

invalid_content = df[
    ~df["explanation_content_valid"]
]

print(
    f"Rows with incomplete explanation content: "
    f"{len(invalid_content)}"
)

if len(invalid_content) > 0:
    print("\nInvalid rows:")
    print(
        invalid_content[
            [
                "occupation_label",
                "skill_label",
                "priority",
            ]
        ].to_string(index=False)
    )

explanation_content_pass = len(invalid_content) == 0

print("PASS" if explanation_content_pass else "FAIL")


# --------------------------------------------------------------
# CHECK 10 — PRIORITY DISTRIBUTION
# --------------------------------------------------------------

print("\n" + "=" * 70)
print("10. PRIORITY DISTRIBUTION")
print("=" * 70)

actual_distribution = (
    df["priority"]
    .value_counts()
    .to_dict()
)

expected_distribution = {
    "High": 4,
    "Moderate": 10,
    "Emerging": 21,
}

print("Actual:")
print(actual_distribution)

print("\nExpected:")
print(expected_distribution)

distribution_pass = (
    actual_distribution == expected_distribution
)

print(
    "\nPASS"
    if distribution_pass
    else "FAIL"
)


# --------------------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------------------

checks = {
    "row_count": row_count_pass,
    "duplicates": duplicates_pass,
    "gap_category": gap_category_pass,
    "market_supported": market_supported_pass,
    "priority_labels": priority_labels_pass,
    "priority_rule": priority_rule_pass,
    "evidence_fields": evidence_fields_pass,
    "explanation_completeness": explanation_completeness_pass,
    "explanation_content": explanation_content_pass,
    "distribution": distribution_pass,
}


print("\n" + "=" * 70)
print("PHASE 6.2 — FINAL VALIDATION SUMMARY")
print("=" * 70)

for name, result in checks.items():
    print(
        f"{name:<28} "
        f"{'PASS' if result else 'FAIL'}"
    )


all_pass = all(checks.values())

print("\n" + "=" * 70)

if all_pass:
    print("PHASE 6.2 VALIDATION: ALL CHECKS PASSED")
else:
    print("PHASE 6.2 VALIDATION: CHECKS FAILED")

print("=" * 70)


if not all_pass:
    raise SystemExit(1)