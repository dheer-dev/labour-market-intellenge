"""Evidence-aware, user-entered skill comparison (no retraining or data writes)."""
from __future__ import annotations

import pandas as pd


def analyze_personal_skills(
    occupation_uri: str,
    selected_skill_uris: set[str],
    relations: pd.DataFrame,
    demand: pd.DataFrame | None = None,
    career: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return ESCO-linked skill evidence; absence of self-report != lack of proficiency."""
    required = {"occupation_uri", "skill_uri", "skill_label"}
    if not required.issubset(relations.columns):
        raise ValueError(f"ESCO relations missing columns: {sorted(required - set(relations.columns))}")
    rows = relations.loc[relations["occupation_uri"].eq(occupation_uri)].copy()
    if rows.empty:
        return pd.DataFrame()
    rows = rows.dropna(subset=["skill_uri"]).drop_duplicates(subset=["skill_uri"]).reset_index(drop=True)
    rows["user_reported"] = rows["skill_uri"].isin(selected_skill_uris)
    rows["potential_gap"] = ~rows["user_reported"]
    rows["status"] = rows["user_reported"].map({True: "Self-reported", False: "Not reported (potential gap)"})
    if demand is not None and {"occupation_uri", "skill_uri"}.issubset(demand.columns):
        fields = [x for x in ["skill_posting_count", "skill_prevalence_percent", "demand_category"] if x in demand]
        d = demand.loc[demand["occupation_uri"].eq(occupation_uri), ["skill_uri"] + fields].copy()
        d = d.drop_duplicates(subset=["skill_uri"])
        rows = rows.merge(d, on="skill_uri", how="left", validate="one_to_one")
        rows["nyc_record_found"] = rows["skill_uri"].isin(set(d["skill_uri"]))
    else:
        rows["nyc_record_found"] = False
    if career is not None and {"occupation_label", "skill_label", "priority"}.issubset(career.columns):
        # Only attach existing priority when the occupation and skill labels uniquely identify a record.
        candidates = career[["occupation_label", "skill_label", "priority"]].dropna().drop_duplicates()
        occ_label = str(rows["occupation_label"].iloc[0]) if "occupation_label" in rows else ""
        candidates = candidates.loc[candidates["occupation_label"].str.casefold() == occ_label.casefold()]
        candidates = candidates.drop(columns=["occupation_label"])
        candidates = candidates.drop_duplicates(subset=["skill_label"], keep=False)
        rows = rows.merge(candidates, on="skill_label", how="left", validate="many_to_one")
    if "priority" not in rows:
        rows["priority"] = pd.NA
    rows["evidence_basis"] = "ESCO occupation–skill relationship"
    rows.loc[rows["nyc_record_found"], "evidence_basis"] += " + NYC snapshot record"
    rows["interpretation"] = "A skill not reported by the user is a potential review item, not a verified lack of proficiency."
    return rows
