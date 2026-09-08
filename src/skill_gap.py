"""
Skill Gap Analysis Module
-------------------------
Calculates discrepancies between market demand skills and candidate/profile skills.
"""

from typing import List, Dict, Set


def analyze_skill_gap(demanded_skills: List[str], current_skills: List[str]) -> Dict[str, List[str]]:
    """Analyze the overlap and missing skills."""
    demand_set: Set[str] = set(demanded_skills)
    current_set: Set[str] = set(current_skills)

    missing_skills = list(demand_set - current_set)
    matching_skills = list(demand_set & current_set)

    return {
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
    }


if __name__ == "__main__":
    print("Skill gap analysis module initialized.")
