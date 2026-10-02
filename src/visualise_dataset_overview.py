import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIGURES_DIR = BASE_DIR / "figures"

FIGURES_DIR.mkdir(exist_ok=True)


def main():

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    esco = pd.read_csv(
        PROCESSED_DIR / "esco_occupation_skill_relations.csv"
    )

    market = pd.read_csv(
        PROCESSED_DIR / "occupation_skill_demand_matrix.csv"
    )

    personal = pd.read_csv(
        PROCESSED_DIR / "personal_skill_gaps.csv"
    )

    # --------------------------------------------------
    # Calculate coverage statistics
    # --------------------------------------------------

    esco_pairs = esco[
        ["occupation_uri", "skill_uri"]
    ].drop_duplicates()

    market_pairs = market[
        ["occupation_uri", "skill_uri"]
    ].drop_duplicates()

    overlap = esco_pairs.merge(
        market_pairs,
        on=["occupation_uri", "skill_uri"],
        how="inner",
    )

    categories = [
        "ESCO occupation-skill pairs",
        "Market-observed pairs",
        "ESCO–market overlap",
        "Personal market-relevant gaps",
    ]

    values = [
        len(esco_pairs),
        len(market_pairs),
        len(overlap),
        len(personal),
    ]

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        categories,
        values,
    )

    plt.title(
        "Labour Market Skill Coverage Overview",
        fontsize=14,
    )

    plt.ylabel(
        "Number of occupation-skill relationships"
    )

    plt.xticks(
        rotation=20,
        ha="right",
    )

    # Add values above bars
    for bar, value in zip(bars, values):

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output = FIGURES_DIR / "dataset_coverage_overview.png"

    plt.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print("=" * 70)
    print("DATASET COVERAGE VISUALISATION CREATED")
    print("=" * 70)

    print("ESCO pairs:", len(esco_pairs))
    print("Market pairs:", len(market_pairs))
    print("Overlap:", len(overlap))
    print("Personal market-relevant gaps:", len(personal))

    print()
    print("Saved:", output)


if __name__ == "__main__":
    main()