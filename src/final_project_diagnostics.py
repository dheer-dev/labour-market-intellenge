
from pathlib import Path
import json
import math

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
REPORT_PATH = DATA / "final_project_diagnostics.csv"

checks = []


def record(name, status, observed, expected, details=""):
    checks.append({
        "check": name,
        "status": status,
        "observed": str(observed),
        "expected": str(expected),
        "details": details,
    })


def read_csv(name, required=True):
    path = DATA / name

    if not path.exists():
        record(
            f"File exists: {name}",
            "FAIL" if required else "WARN",
            "missing",
            "present" if required else "optional",
            "Required file missing" if required else "Optional file not found",
        )
        return None

    try:
        df = pd.read_csv(path)
        record(
            f"File readable: {name}",
            "PASS",
            f"{len(df)} rows",
            "readable CSV",
        )
        return df
    except Exception as exc:
        record(f"File readable: {name}", "FAIL", str(exc), "readable CSV")
        return None


def check_columns(df, filename, required_columns):
    if df is None:
        return False

    missing = sorted(set(required_columns) - set(df.columns))
    record(
        f"Required columns: {filename}",
        "PASS" if not missing else "FAIL",
        "none missing" if not missing else missing,
        "all required columns present",
    )
    return not missing


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 1. O*NET INPUT AND REVIEW
    # -----------------------------------------------------

    onet = read_csv("onet_target_occupation_skills.csv")
    review = read_csv("onet_esco_skill_mapping_initial_review.csv")
    esco = read_csv("esco_occupation_skill_relations.csv")

    if onet is not None:
        check_columns(
            onet,
            "onet_target_occupation_skills.csv",
            [
                "target_occupation", "onet_soc_code", "element_id",
                "element_name", "onet_skill_category",
                "onet_importance", "onet_level",
            ],
        )

        record(
            "O*NET skill records",
            "PASS" if len(onet) == 140 else "WARN",
            len(onet),
            "140 expected from current four-occupation extraction",
        )

        duplicate_count = int(
            onet.duplicated(
                ["target_occupation", "element_id"]
            ).sum()
        )
        record(
            "O*NET occupation-skill duplicates",
            "PASS" if duplicate_count == 0 else "FAIL",
            duplicate_count,
            0,
        )

        record(
            "O*NET occupation coverage",
            "PASS" if onet["target_occupation"].nunique() == 4 else "WARN",
            onet["target_occupation"].nunique(),
            4,
            ", ".join(sorted(onet["target_occupation"].dropna().unique())),
        )

        for col in ["onet_importance", "onet_level"]:
            values = pd.to_numeric(onet[col], errors="coerce")
            missing = int(values.isna().sum())
            record(
                f"O*NET {col} numeric coverage",
                "PASS" if missing == 0 else "WARN",
                f"{len(values) - missing}/{len(values)} numeric",
                "all rows numeric",
            )

    if review is not None:
        check_columns(
            review,
            "onet_esco_skill_mapping_initial_review.csv",
            [
                "onet_element_id", "onet_skill_label",
                "review_decision", "reviewed_esco_skill_uri",
                "reviewed_esco_skill_label",
            ],
        )

        unique_skills = review["onet_element_id"].nunique()
        record(
            "Unique reviewed O*NET elements",
            "PASS" if unique_skills == 35 else "WARN",
            unique_skills,
            35,
        )

        valid_decisions = {
            "APPROVED", "REJECTED",
            "NO_DIRECT_MATCH", "REVIEW_REQUIRED",
        }
        invalid = sorted(
            set(review["review_decision"].dropna().unique())
            - valid_decisions
        )
        record(
            "Review decision validity",
            "PASS" if not invalid else "FAIL",
            invalid or "all valid",
            sorted(valid_decisions),
        )

        approved = review["review_decision"].eq("APPROVED")
        approved_missing_uri = int(
            (approved & review["reviewed_esco_skill_uri"].isna()).sum()
        )
        record(
            "Approved mappings have URIs",
            "PASS" if approved_missing_uri == 0 else "FAIL",
            approved_missing_uri,
            0,
        )

        decision_counts = (
            review["review_decision"].value_counts().to_dict()
        )
        for decision in sorted(valid_decisions):
            record(
                f"Review count: {decision}",
                "INFO",
                decision_counts.get(decision, 0),
                "informational",
            )

    # -----------------------------------------------------
    # 2. ESCO CROSSWALK
    # -----------------------------------------------------

    crosswalk = read_csv("onet_esco_skill_crosswalk.csv")
    occupation_skills = read_csv("onet_target_skills_with_crosswalk.csv")
    esco_uris = set()

    if esco is not None:
        check_columns(
            esco,
            "esco_occupation_skill_relations.csv",
            ["occupation_uri", "skill_uri", "skill_label", "relation_type"],
        )
        esco_uris = set(esco["skill_uri"].dropna().astype(str))

    if crosswalk is not None:
        check_columns(
            crosswalk,
            "onet_esco_skill_crosswalk.csv",
            [
                "onet_element_id", "review_decision",
                "reviewed_esco_skill_uri", "mapping_usable",
            ],
        )

        record(
            "Crosswalk rows",
            "PASS" if len(crosswalk) == 35 else "WARN",
            len(crosswalk),
            35,
        )

        duplicate_elements = int(
            crosswalk["onet_element_id"].duplicated().sum()
        )
        record(
            "Crosswalk duplicate elements",
            "PASS" if duplicate_elements == 0 else "FAIL",
            duplicate_elements,
            0,
        )

        approved = crosswalk["review_decision"].eq("APPROVED")
        approved_uris = crosswalk.loc[
            approved, "reviewed_esco_skill_uri"
        ].dropna().astype(str)

        invalid_approved_uris = [
            uri for uri in approved_uris if uri not in esco_uris
        ]

        record(
            "Approved URIs found in ESCO relations",
            "PASS" if not invalid_approved_uris else "WARN",
            len(approved_uris) - len(invalid_approved_uris),
            f"{len(approved_uris)} approved URIs present",
            "Presence confirms dataset coverage, not semantic equivalence.",
        )

    if occupation_skills is not None:
        record(
            "Occupation-level O*NET records",
            "PASS" if len(occupation_skills) == 140 else "WARN",
            len(occupation_skills),
            140,
        )

        if "review_decision" in occupation_skills.columns:
            missing_decisions = int(
                occupation_skills["review_decision"].isna().sum()
            )
            record(
                "Occupation records with review decisions",
                "PASS" if missing_decisions == 0 else "FAIL",
                len(occupation_skills) - missing_decisions,
                f"{len(occupation_skills)}",
            )

    # -----------------------------------------------------
    # 3. EXISTING CAREER DECISION SUPPORT AND ANALYTICS
    # -----------------------------------------------------

    expected_columns = {
        "occupation_skill_demand_matrix.csv": [
            "occupation_uri", "occupation_label", "skill_uri", "skill_label",
        ],
        "occupation_skill_demand_analysis.csv": [
            "occupation_label", "skill_label",
        ],
        "occupation_skill_gap_analysis.csv": [
            "occupation_label", "skill_label",
        ],
        "skill_gap_prioritisation.csv": [
            "occupation_label", "skill_label",
        ],
        "skill_gap_explanations.csv": [
            "occupation_label", "skill_label",
        ],
        "career_decision_support.csv": [
            "occupation_label", "skill_label", "priority",
        ],
        "personal_skill_gaps.csv": [
            "occupation_label", "skill_label",
        ],
        "user_occupation_skill_comparison.csv": [
            "occupation_label", "skill_label",
        ],
    }

    for filename, required in expected_columns.items():
        df = read_csv(filename, required=False)

        if df is None:
            continue

        if not check_columns(df, filename, required):
            continue

        if {"occupation_label", "skill_label"}.issubset(df.columns):
            null_keys = int(
                df[["occupation_label", "skill_label"]].isna().any(axis=1).sum()
            )
            record(
                f"Missing occupation/skill keys: {filename}",
                "PASS" if null_keys == 0 else "WARN",
                null_keys,
                0,
            )

    career = read_csv("career_decision_support.csv", required=False)

    if career is not None and "priority" in career.columns:
        record(
            "Career decision-support rows",
            "INFO",
            len(career),
            "existing output retained",
        )
        record(
            "Career decision-support priority distribution",
            "INFO",
            career["priority"].value_counts(dropna=False).to_dict(),
            "inspect distribution; no new thresholds imposed",
        )

    # API JSON integrity, if available.
    api_path = DATA / "career_decision_support_api.json"
    if api_path.exists():
        try:
            with api_path.open(encoding="utf-8") as f:
                api_data = json.load(f)
            record(
                "Career decision-support API JSON",
                "PASS",
                "valid JSON",
                "valid JSON",
                f"Root type: {type(api_data).__name__}",
            )
        except Exception as exc:
            record(
                "Career decision-support API JSON",
                "FAIL",
                str(exc),
                "valid JSON",
            )
    else:
        record(
            "Career decision-support API JSON",
            "WARN",
            "missing",
            "optional existing artifact",
        )

    # -----------------------------------------------------
    # 4. ML EVIDENCE CHECK
    # -----------------------------------------------------
    # Inspect common saved evaluation artifact names if present.
    # Do not rerun training or infer metrics from model files.

    ml_names = [
        "model_evaluation.csv",
        "ml_model_evaluation.csv",
        "model_metrics.csv",
        "cross_validation_results.csv",
        "ml_evaluation_results.csv",
    ]

    found_ml = []
    for name in ml_names:
        path = DATA / name
        if path.exists():
            found_ml.append(name)
            ml_df = read_csv(name, required=False)
            if ml_df is not None:
                record(
                    f"ML evaluation artifact: {name}",
                    "INFO",
                    f"{len(ml_df)} rows",
                    "inspect saved evaluation",
                    "This script does not retrain the model.",
                )

    record(
        "Saved ML evaluation file discovery",
        "INFO" if found_ml else "WARN",
        found_ml or "none of the standard filenames found",
        "review existing evaluation output if available",
        "Absence here does not mean the model was not evaluated.",
    )

    # -----------------------------------------------------
    # 5. FINAL REPORT
    # -----------------------------------------------------

    report = pd.DataFrame(checks)
    report.to_csv(REPORT_PATH, index=False)

    failures = int(report["status"].eq("FAIL").sum())
    warnings = int(report["status"].eq("WARN").sum())
    passed = int(report["status"].eq("PASS").sum())

    print("\n" + "=" * 68)
    print("FINAL PROJECT DIAGNOSTICS")
    print("=" * 68)
    print(f"Checks passed: {passed}")
    print(f"Warnings:      {warnings}")
    print(f"Failures:      {failures}")
    print(f"\nFull report: {REPORT_PATH}")

    if failures:
        print("\nChecks requiring fixes:")
        print(
            report.loc[
                report["status"].eq("FAIL"),
                ["check", "observed", "expected", "details"],
            ].to_string(index=False)
        )

    if warnings:
        print("\nWarnings to review:")
        print(
            report.loc[
                report["status"].eq("WARN"),
                ["check", "observed", "expected"],
            ].to_string(index=False)
        )

    print("\nNo training, remapping, or existing analysis outputs were changed.")
    print(
        "Reminder: the O*NET–ESCO crosswalk is preliminary; "
        "unresolved skills must not be presented as matched."
    )


if __name__ == "__main__":
    main()
