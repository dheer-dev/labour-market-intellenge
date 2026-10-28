import pandas as pd
from pathlib import Path


RANDOM_STATE = 42
NEGATIVE_RATIO = 3

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ESCO_FILE = PROCESSED_DIR / "esco_occupation_skill_relations.csv"
MARKET_FILE = PROCESSED_DIR / "occupation_skill_demand_matrix.csv"

OUTPUT_FILE = PROCESSED_DIR / "model_dataset_occupation_aware.csv"
OCCUPATION_SUMMARY_FILE = (
    PROCESSED_DIR / "model_dataset_occupation_aware_by_occupation.csv"
)
SUMMARY_FILE = (
    PROCESSED_DIR / "model_dataset_occupation_aware_summary.csv"
)


def main():
    print("=" * 70)
    print("BUILDING OCCUPATION-AWARE MODEL DATASET")
    print("=" * 70)

    # 1. Load source datasets
    esco = pd.read_csv(ESCO_FILE)
    market = pd.read_csv(MARKET_FILE)

    required_esco = {
        "occupation_uri",
        "occupation_label",
        "skill_uri",
        "skill_label",
        "relation_type",
        "skill_type",
    }
    required_market = {"occupation_uri", "skill_uri"}

    if not required_esco.issubset(esco.columns):
        raise ValueError(
            f"ESCO file missing columns: "
            f"{sorted(required_esco - set(esco.columns))}"
        )

    if not required_market.issubset(market.columns):
        raise ValueError(
            f"Market file missing columns: "
            f"{sorted(required_market - set(market.columns))}"
        )

    # 2. Construct unique ESCO occupation-skill pairs
    esco_pairs = esco[
        [
            "occupation_uri",
            "occupation_label",
            "skill_uri",
            "skill_label",
            "relation_type",
            "skill_type",
        ]
    ].drop_duplicates(
        subset=["occupation_uri", "skill_uri"]
    ).copy()

    # 3. Identify pairs observed in the market
    market_pairs = market[
        ["occupation_uri", "skill_uri"]
    ].drop_duplicates().copy()

    market_pairs["market_observed"] = 1

    candidates = esco_pairs.merge(
        market_pairs,
        on=["occupation_uri", "skill_uri"],
        how="left",
        validate="one_to_one",
    )

    candidates["market_observed"] = (
        candidates["market_observed"]
        .fillna(0)
        .astype(int)
    )

    if candidates.duplicated(
        subset=["occupation_uri", "skill_uri"]
    ).any():
        raise ValueError("Duplicate candidate occupation-skill pairs found.")

    observed = candidates[
        candidates["market_observed"] == 1
    ].copy()

    non_observed = candidates[
        candidates["market_observed"] == 0
    ].copy()

    print("Full ESCO pairs:", len(candidates))
    print("Observed pairs:", len(observed))
    print("Available non-observed pairs:", len(non_observed))

    # 4. Sample non-observed pairs within each occupation.
    # All observed pairs are retained.
    sampled_groups = []

    positive_counts = observed.groupby("occupation_uri").size()

    negative_groups = {
        occupation_uri: group
        for occupation_uri, group in non_observed.groupby(
            "occupation_uri", sort=True
        )
    }

    for occupation_uri, positive_count in positive_counts.sort_index().items():
        available = negative_groups.get(occupation_uri)

        if available is None or available.empty:
            continue

        sample_size = min(
            len(available),
            int(positive_count) * NEGATIVE_RATIO,
        )

        sampled_groups.append(
            available.sample(
                n=sample_size,
                random_state=RANDOM_STATE,
            )
        )

    if not sampled_groups:
        raise ValueError("No eligible non-observed pairs could be sampled.")

    sampled_non_observed = pd.concat(
        sampled_groups,
        ignore_index=True,
    )

    # 5. Combine and deterministically shuffle
    model_dataset = pd.concat(
        [observed, sampled_non_observed],
        ignore_index=True,
    )

    model_dataset = model_dataset.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # 6. Validate dataset integrity
    duplicate_count = model_dataset.duplicated(
        subset=["occupation_uri", "skill_uri"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            f"Duplicate occupation-skill pairs: {duplicate_count}"
        )

    if model_dataset["market_observed"].isna().any():
        raise ValueError("Missing market_observed labels found.")

    if set(model_dataset["market_observed"].unique()) != {0, 1}:
        raise ValueError("Dataset must contain both classes.")

    # Every included occupation must have both classes.
    class_counts = model_dataset.groupby(
        "occupation_uri"
    )["market_observed"].nunique()

    if not (class_counts == 2).all():
        raise ValueError(
            "At least one included occupation does not have both classes."
        )

    # 7. Build per-occupation audit
    by_occupation = model_dataset.groupby(
        ["occupation_uri", "occupation_label"],
        as_index=False,
    ).agg(
        total_pairs=("market_observed", "size"),
        observed_pairs=("market_observed", "sum"),
    )

    by_occupation["non_observed_pairs"] = (
        by_occupation["total_pairs"]
        - by_occupation["observed_pairs"]
    )

    by_occupation["observed_percent"] = (
        100 * by_occupation["observed_pairs"]
        / by_occupation["total_pairs"]
    ).round(2)

    # 8. Build overall audit summary
    total_observed = int(
        (model_dataset["market_observed"] == 1).sum()
    )
    total_non_observed = int(
        (model_dataset["market_observed"] == 0).sum()
    )

    summary = pd.DataFrame(
        [
            {
                "random_state": RANDOM_STATE,
                "negative_ratio": NEGATIVE_RATIO,
                "full_esco_pairs": len(candidates),
                "full_observed_pairs": len(observed),
                "full_non_observed_pairs": len(non_observed),
                "dataset_rows": len(model_dataset),
                "dataset_observed_pairs": total_observed,
                "dataset_non_observed_pairs": total_non_observed,
                "observed_percent": round(
                    100 * total_observed / len(model_dataset), 2
                ),
                "occupations_in_dataset": (
                    model_dataset["occupation_uri"].nunique()
                ),
                "duplicate_pairs": int(duplicate_count),
                "sampling_method": (
                    "All observed pairs; up to 3 non-observed pairs "
                    "per observed pair within each eligible occupation"
                ),
            }
        ]
    )

    # 9. Save only to new filenames
    model_dataset.to_csv(OUTPUT_FILE, index=False)
    by_occupation.to_csv( OCCUPATION_SUMMARY_FILE, index=False)
    summary.to_csv(SUMMARY_FILE, index=False)

    # 10. Report
    print("\n" + "=" * 70)
    print("OCCUPATION-AWARE DATASET SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    print("\nClass distribution:")
    print(model_dataset["market_observed"].value_counts().sort_index())

    print("\nOccupations with fewest observed pairs:")
    print(
        by_occupation.sort_values(
            ["observed_pairs", "total_pairs"]
        ).head(10).to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_FILE)
    print(OCCUPATION_SUMMARY_FILE)
    print(SUMMARY_FILE)
    print("\nOriginal dataset and model files were not modified.")


if __name__ == "__main__":
    main()
