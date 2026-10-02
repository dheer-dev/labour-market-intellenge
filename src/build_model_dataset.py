import pandas as pd
from pathlib import Path


RANDOM_STATE = 42
NEGATIVE_RATIO = 3


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"


ESCO_FILE = PROCESSED_DIR / "esco_occupation_skill_relations.csv"
MARKET_FILE = PROCESSED_DIR / "occupation_skill_demand_matrix.csv"
OUTPUT_FILE = PROCESSED_DIR / "model_dataset.csv"


def main():

    print("=" * 70)
    print("BUILDING MODEL DATASET")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------

    esco = pd.read_csv(ESCO_FILE)
    market = pd.read_csv(MARKET_FILE)

    print("ESCO relations:", len(esco))
    print("Market demand rows:", len(market))

    # --------------------------------------------------
    # 2. Keep unique ESCO occupation-skill relations
    # --------------------------------------------------

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
    )

    # --------------------------------------------------
    # 3. Identify occupation-skill pairs observed
    #    in the NYC market
    # --------------------------------------------------

    market_pairs = market[
        [
            "occupation_uri",
            "skill_uri",
        ]
    ].drop_duplicates()

    market_pairs["market_observed"] = 1

    dataset = esco_pairs.merge(
        market_pairs,
        on=["occupation_uri", "skill_uri"],
        how="left",
    )

    dataset["market_observed"] = (
        dataset["market_observed"]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------
    # 4. Separate observed and non-observed pairs
    # --------------------------------------------------

    observed = dataset[
        dataset["market_observed"] == 1
    ].copy()

    non_observed = dataset[
        dataset["market_observed"] == 0
    ].copy()

    print()
    print("Observed pairs:", len(observed))
    print("Non-observed pairs:", len(non_observed))

    # --------------------------------------------------
    # 5. Sample non-observed pairs
    # --------------------------------------------------

    sample_size = min(
        len(non_observed),
        len(observed) * NEGATIVE_RATIO
    )

    sampled_non_observed = non_observed.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------
    # 6. Combine
    # --------------------------------------------------

    model_dataset = pd.concat(
        [
            observed,
            sampled_non_observed,
        ],
        ignore_index=True,
    )

    # Shuffle deterministically
    model_dataset = model_dataset.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 7. Validation
    # --------------------------------------------------

    duplicate_count = model_dataset.duplicated(
        subset=["occupation_uri", "skill_uri"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Duplicate occupation-skill pairs found: "
            f"{duplicate_count}"
        )

    if model_dataset["market_observed"].isna().any():
        raise ValueError(
            "Missing values found in market_observed."
        )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------

    model_dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------
    # 9. Report
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL DATASET SUMMARY")
    print("=" * 70)

    print("Output:", OUTPUT_FILE)
    print("Rows:", len(model_dataset))
    print("Columns:", len(model_dataset.columns))
    print("Observed:", (model_dataset["market_observed"] == 1).sum())
    print(
        "Non-observed:",
        (model_dataset["market_observed"] == 0).sum(),
    )
    print(
        "Observed rate:",
        round(
            model_dataset["market_observed"].mean() * 100,
            2,
        ),
        "%",
    )
    print("Duplicate pairs:", duplicate_count)

    print()
    print("Class distribution:")
    print(
        model_dataset["market_observed"]
        .value_counts()
        .sort_index()
    )

    print("=" * 70)
    print("MODEL DATASET CREATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()