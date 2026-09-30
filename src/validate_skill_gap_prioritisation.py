from pathlib import Path

import pandas as pd


# ============================================================
# PHASE 5 — SKILL-GAP ENGINE
# Step 5: Final Prioritisation Validation
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "occupation_skill_gap_analysis.csv"
)

PRIORITY_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "skill_gap_prioritisation.csv"
)


print("=" * 70)
print("PHASE 5 — FINAL PRIORITISATION VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. Load source and final prioritisation outputs
# ------------------------------------------------------------

source = pd.read_csv(INPUT_FILE)
priority = pd.read_csv(PRIORITY_FILE)

print("\nSource shape:", source.shape)
print("Prioritisation shape:", priority.shape)


# ------------------------------------------------------------
# 2. Reconstruct the expected candidate population
# ------------------------------------------------------------

expected = source[
    (source["gap_category"] == "Market-only")
    & (source["market_supported"] == True)
].copy()

print("\nExpected candidate gaps:", len(expected))
print("Final prioritised gaps:", len(priority))


# ------------------------------------------------------------
# 3. Candidate-count validation
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. CANDIDATE COUNT VALIDATION")
print("=" * 70)

count_match = len(expected) == len(priority)

print("Expected:", len(expected))
print("Actual:  ", len(priority))
print("PASS" if count_match else "FAIL")


# ------------------------------------------------------------
# 4. Validate unique occupation-skill pairs
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. DUPLICATE VALIDATION")
print("=" * 70)

source_duplicates = expected.duplicated(
    subset=["occupation_label", "skill_label"]
).sum()

priority_duplicates = priority.duplicated(
    subset=["occupation_label", "skill_label"]
).sum()

print("Source candidate duplicates:", source_duplicates)
print("Final output duplicates:   ", priority_duplicates)

if source_duplicates == 0 and priority_duplicates == 0:
    print("PASS")
else:
    print("FAIL")


# ------------------------------------------------------------
# 5. Verify that no non-candidate gaps entered the output
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. CANDIDATE MEMBERSHIP VALIDATION")
print("=" * 70)

expected_keys = set(
    zip(
        expected["occupation_label"],
        expected["skill_label"],
    )
)

priority_keys = set(
    zip(
        priority["occupation_label"],
        priority["skill_label"],
    )
)

unexpected = priority_keys - expected_keys
missing = expected_keys - priority_keys

print("Unexpected final gaps:", len(unexpected))
print("Missing candidate gaps:", len(missing))

if len(unexpected) == 0 and len(missing) == 0:
    print("PASS")
else:
    print("FAIL")

if unexpected:
    print("\nUnexpected:")
    for item in sorted(unexpected):
        print(item)

if missing:
    print("\nMissing:")
    for item in sorted(missing):
        print(item)


# ------------------------------------------------------------
# 6. Validate priority labels
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. PRIORITY LABEL VALIDATION")
print("=" * 70)

valid_priorities = {
    "High",
    "Moderate",
    "Emerging",
}

invalid_priorities = set(
    priority["priority"].dropna().unique()
) - valid_priorities

print("Invalid priority labels:", invalid_priorities)

if len(invalid_priorities) == 0:
    print("PASS")
else:
    print("FAIL")


# ------------------------------------------------------------
# 7. Recalculate expected priorities
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. PRIORITY RULE REPRODUCIBILITY")
print("=" * 70)


def expected_priority(prevalence):
    if prevalence >= 40:
        return "High"
    elif prevalence >= 30:
        return "Moderate"
    else:
        return "Emerging"


expected["expected_priority"] = (
    expected["market_skill_prevalence_percent"]
    .apply(expected_priority)
)


expected_lookup = {
    (
        row["occupation_label"],
        row["skill_label"],
    ): row["expected_priority"]
    for _, row in expected.iterrows()
}


priority["expected_priority"] = priority.apply(
    lambda row: expected_lookup.get(
        (
            row["occupation_label"],
            row["skill_label"],
        )
    ),
    axis=1,
)

priority_mismatches = priority[
    priority["priority"] != priority["expected_priority"]
].copy()

print("Priority mismatches:", len(priority_mismatches))

if len(priority_mismatches) == 0:
    print("PASS")
else:
    print("FAIL")
    print(
        priority_mismatches[
            [
                "occupation_label",
                "skill_label",
                "market_skill_prevalence_percent",
                "priority",
                "expected_priority",
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 8. Validate threshold boundaries explicitly
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. THRESHOLD BOUNDARY VALIDATION")
print("=" * 70)

boundary_tests = pd.DataFrame(
    {
        "prevalence": [
            20.00,
            20.01,
            29.99,
            30.00,
            30.01,
            39.99,
            40.00,
            40.01,
        ]
    }
)

boundary_tests["expected_priority"] = (
    boundary_tests["prevalence"]
    .apply(expected_priority)
)

print(boundary_tests.to_string(index=False))

expected_boundary_labels = [
    "Emerging",
    "Emerging",
    "Emerging",
    "Moderate",
    "Moderate",
    "Moderate",
    "High",
    "High",
]

boundary_pass = (
    boundary_tests["expected_priority"].tolist()
    == expected_boundary_labels
)

print("\nBoundary rule:", "PASS" if boundary_pass else "FAIL")


# ------------------------------------------------------------
# 9. Validate prevalence range
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. PREVALENCE RANGE VALIDATION")
print("=" * 70)

invalid_prevalence = priority[
    (
        priority["market_skill_prevalence_percent"] < 0
    )
    |
    (
        priority["market_skill_prevalence_percent"] > 100
    )
]

print(
    "Invalid prevalence values:",
    len(invalid_prevalence),
)

if len(invalid_prevalence) == 0:
    print("PASS")
else:
    print("FAIL")


# ------------------------------------------------------------
# 10. Validate explanations
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. EXPLANATION COMPLETENESS")
print("=" * 70)

required_explanation_fields = priority[
    [
        "priority",
        "market_skill_prevalence_percent",
        "market_skill_rank",
        "market_demand_category",
        "priority_explanation",
    ]
]

missing_explanations = (
    required_explanation_fields.isna().any(axis=1)
).sum()

empty_explanations = (
    priority["priority_explanation"]
    .astype(str)
    .str.strip()
    .eq("")
).sum()

print("Rows with missing explanation fields:", missing_explanations)
print("Rows with empty explanations:", empty_explanations)

if missing_explanations == 0 and empty_explanations == 0:
    print("PASS")
else:
    print("FAIL")


# ------------------------------------------------------------
# 11. Validate priority distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. PRIORITY DISTRIBUTION")
print("=" * 70)

distribution = (
    priority["priority"]
    .value_counts()
    .reindex(
        ["High", "Moderate", "Emerging"],
        fill_value=0,
    )
)

print(distribution)

expected_distribution = pd.Series(
    [4, 10, 21],
    index=["High", "Moderate", "Emerging"],
)

distribution_pass = distribution.equals(
    expected_distribution
)

print(
    "\nExpected distribution:",
    expected_distribution.to_dict(),
)

print(
    "Distribution check:",
    "PASS" if distribution_pass else "FAIL"
)


# ------------------------------------------------------------
# 12. Final validation summary
# ------------------------------------------------------------

checks = {
    "candidate_count": count_match,
    "duplicates": (
        source_duplicates == 0
        and priority_duplicates == 0
    ),
    "candidate_membership": (
        len(unexpected) == 0
        and len(missing) == 0
    ),
    "priority_labels": (
        len(invalid_priorities) == 0
    ),
    "rule_reproducibility": (
        len(priority_mismatches) == 0
    ),
    "threshold_boundaries": boundary_pass,
    "prevalence_range": (
        len(invalid_prevalence) == 0
    ),
    "explanations": (
        missing_explanations == 0
        and empty_explanations == 0
    ),
    "distribution": distribution_pass,
}


print("\n" + "=" * 70)
print("10. FINAL VALIDATION SUMMARY")
print("=" * 70)

for check, result in checks.items():
    print(
        f"{check:<30}",
        "PASS" if result else "FAIL"
    )

all_pass = all(checks.values())

print("\n" + "=" * 70)

if all_pass:
    print("PHASE 5 VALIDATION: ALL CHECKS PASSED")
else:
    print("PHASE 5 VALIDATION: REVIEW REQUIRED")

print("=" * 70)