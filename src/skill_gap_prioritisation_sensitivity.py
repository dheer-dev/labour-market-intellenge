
from pathlib import Path

import pandas as pd


# ============================================================
# PHASE 9 — ROBUSTNESS AND SENSITIVITY ANALYSIS
# Test the stability of skill-gap priority classifications.
# The existing prioritisation pipeline remains unchanged.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "occupation_skill_gap_analysis.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

ASSIGNMENTS_FILE = (
    OUTPUT_DIR / "skill_gap_sensitivity_assignments.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "skill_gap_sensitivity_summary.csv"
)

TRANSITIONS_FILE = (
    OUTPUT_DIR / "skill_gap_sensitivity_transitions.csv"
)

OCCUPATION_FILE = (
    OUTPUT_DIR / "skill_gap_sensitivity_by_occupation.csv"
)

# Current thresholds used by the existing final script.
BASELINE_HIGH = 40
BASELINE_MODERATE = 30

# Alternative threshold values for sensitivity testing.
HIGH_THRESHOLDS = [35, 40, 45]
MODERATE_THRESHOLDS = [25, 30, 35]

PRIORITY_ORDER = ["High", "Moderate", "Emerging"]


def assign_priority(prevalence, high_threshold, moderate_threshold):
    """Apply the existing three-level priority rule."""

    if prevalence >= high_threshold:
        return "High"
    elif prevalence >= moderate_threshold:
        return "Moderate"
    return "Emerging"


def main():
    print("=" * 70)
    print("SKILL-GAP PRIORITISATION SENSITIVITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load data and apply the existing candidate filter
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "occupation_label",
        "skill_label",
        "gap_category",
        "market_supported",
        "market_skill_prevalence_percent",
    ]

    missing = sorted(set(required_columns) - set(df.columns))

    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}"
        )

    candidates = df[
        (df["gap_category"] == "Market-only")
        & (df["market_supported"] == True)
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No candidate gaps found. Check gap_category and "
            "market_supported values in the input."
        )

    candidates["market_skill_prevalence_percent"] = pd.to_numeric(
        candidates["market_skill_prevalence_percent"],
        errors="raise",
    )

    if candidates["market_skill_prevalence_percent"].isna().any():
        raise ValueError(
            "Candidate gaps contain missing prevalence values."
        )

    # A stable key lets us compare the same occupation-skill pair
    # across scenarios without changing the source data.
    candidates = candidates.reset_index(drop=True)
    candidates["gap_id"] = (
        candidates["occupation_label"].astype(str)
        + " || "
        + candidates["skill_label"].astype(str)
    )

    if candidates["gap_id"].duplicated().any():
        duplicates = candidates.loc[
            candidates["gap_id"].duplicated(keep=False), "gap_id"
        ].tolist()
        raise ValueError(
            "Duplicate occupation-skill candidate pairs found: "
            f"{duplicates[:10]}"
        )

    print(f"\nInput rows: {len(df)}")
    print(f"Candidate gaps: {len(candidates)}")
    print(
        "Baseline thresholds: "
        f"High >= {BASELINE_HIGH}%; "
        f"Moderate >= {BASELINE_MODERATE}%"
    )

    # --------------------------------------------------------
    # 2. Build the baseline and alternative scenarios
    # --------------------------------------------------------

    scenarios = []

    for high_threshold in HIGH_THRESHOLDS:
        for moderate_threshold in MODERATE_THRESHOLDS:
            if moderate_threshold >= high_threshold:
                continue

            scenarios.append(
                {
                    "high_threshold": high_threshold,
                    "moderate_threshold": moderate_threshold,
                    "scenario": (
                        f"High_{high_threshold}_"
                        f"Moderate_{moderate_threshold}"
                    ),
                    "is_baseline": (
                        high_threshold == BASELINE_HIGH
                        and moderate_threshold == BASELINE_MODERATE
                    ),
                }
            )

    if not any(s["is_baseline"] for s in scenarios):
        raise ValueError(
            "Baseline 40/30 configuration is missing from scenarios."
        )

    # --------------------------------------------------------
    # 3. Assign priorities for every scenario
    # --------------------------------------------------------

    assignment_frames = []
    summary_records = []
    occupation_records = []

    for scenario in scenarios:
        high_threshold = scenario["high_threshold"]
        moderate_threshold = scenario["moderate_threshold"]
        scenario_name = scenario["scenario"]

        scenario_df = candidates[
            [
                "gap_id",
                "occupation_label",
                "skill_label",
                "market_skill_prevalence_percent",
            ]
        ].copy()

        scenario_df["scenario"] = scenario_name
        scenario_df["high_threshold"] = high_threshold
        scenario_df["moderate_threshold"] = moderate_threshold
        scenario_df["is_baseline"] = scenario["is_baseline"]

        scenario_df["priority"] = scenario_df[
            "market_skill_prevalence_percent"
        ].apply(
            lambda value: assign_priority(
                value,
                high_threshold,
                moderate_threshold,
            )
        )

        assignment_frames.append(scenario_df)

        counts = (
            scenario_df["priority"]
            .value_counts()
            .reindex(PRIORITY_ORDER, fill_value=0)
        )

        summary_records.append(
            {
                "scenario": scenario_name,
                "high_threshold": high_threshold,
                "moderate_threshold": moderate_threshold,
                "is_baseline": scenario["is_baseline"],
                "candidate_count": len(scenario_df),
                "high_count": int(counts["High"]),
                "moderate_count": int(counts["Moderate"]),
                "emerging_count": int(counts["Emerging"]),
                "high_percent": round(
                    counts["High"] / len(scenario_df) * 100, 2
                ),
                "moderate_percent": round(
                    counts["Moderate"] / len(scenario_df) * 100, 2
                ),
                "emerging_percent": round(
                    counts["Emerging"] / len(scenario_df) * 100, 2
                ),
            }
        )

        occupation_counts = pd.crosstab(
            scenario_df["occupation_label"],
            scenario_df["priority"],
        ).reindex(columns=PRIORITY_ORDER, fill_value=0)

        for occupation, row in occupation_counts.iterrows():
            occupation_records.append(
                {
                    "scenario": scenario_name,
                    "occupation_label": occupation,
                    "high_count": int(row["High"]),
                    "moderate_count": int(row["Moderate"]),
                    "emerging_count": int(row["Emerging"]),
                }
            )

    assignments = pd.concat(
        assignment_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(summary_records)
    by_occupation = pd.DataFrame(occupation_records)

    # --------------------------------------------------------
    # 4. Compare every scenario against the baseline
    # --------------------------------------------------------

    baseline = assignments[
        assignments["is_baseline"]
    ][["gap_id", "priority"]].rename(
        columns={"priority": "baseline_priority"}
    )

    assignments = assignments.merge(
        baseline,
        on="gap_id",
        how="left",
        validate="many_to_one",
    )

    assignments["priority_changed"] = (
        assignments["priority"] != assignments["baseline_priority"]
    )

    change_records = []

    for scenario in scenarios:
        scenario_name = scenario["scenario"]

        current = assignments[
            assignments["scenario"] == scenario_name
        ]

        changed_count = int(current["priority_changed"].sum())

        change_records.append(
            {
                "scenario": scenario_name,
                "is_baseline": scenario["is_baseline"],
                "candidate_count": len(current),
                "changed_gap_count": changed_count,
                "unchanged_gap_count": len(current) - changed_count,
                "changed_percent": round(
                    changed_count / len(current) * 100, 2
                ),
            }
        )

    changes = pd.DataFrame(change_records)

    summary = summary.merge(
        changes[
            [
                "scenario",
                "changed_gap_count",
                "unchanged_gap_count",
                "changed_percent",
            ]
        ],
        on="scenario",
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # 5. Record transitions between baseline and each scenario
    # --------------------------------------------------------

    transition_records = []

    for scenario in scenarios:
        current = assignments[
            assignments["scenario"] == scenario["scenario"]
        ]

        transition_table = pd.crosstab(
            current["baseline_priority"],
            current["priority"],
        ).reindex(
            index=PRIORITY_ORDER,
            columns=PRIORITY_ORDER,
            fill_value=0,
        )

        for baseline_priority in PRIORITY_ORDER:
            for new_priority in PRIORITY_ORDER:
                transition_records.append(
                    {
                        "scenario": scenario["scenario"],
                        "baseline_priority": baseline_priority,
                        "scenario_priority": new_priority,
                        "gap_count": int(
                            transition_table.loc[
                                baseline_priority, new_priority
                            ]
                        ),
                    }
                )

    transitions = pd.DataFrame(transition_records)

    # --------------------------------------------------------
    # 6. Save separate sensitivity outputs
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(ASSIGNMENTS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)
    transitions.to_csv(TRANSITIONS_FILE, index=False)
    by_occupation.to_csv(OCCUPATION_FILE, index=False)

    # --------------------------------------------------------
    # 7. Report results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PRIORITY DISTRIBUTION BY SCENARIO")
    print("=" * 70)

    print(
        summary[
            [
                "scenario",
                "high_count",
                "moderate_count",
                "emerging_count",
                "changed_gap_count",
                "changed_percent",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)

    print(
        summary[
            [
                "scenario",
                "changed_gap_count",
                "unchanged_gap_count",
                "changed_percent",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    for path in [
        ASSIGNMENTS_FILE,
        SUMMARY_FILE,
        TRANSITIONS_FILE,
        OCCUPATION_FILE,
    ]:
        print(path)

    print("\nSensitivity analysis complete.")
    print(
        "The existing skill_gap_prioritisation.csv and "
        "prioritisation script were not modified."
    )


if __name__ == "__main__":
    main()
