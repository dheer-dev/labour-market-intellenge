from pathlib import Path
import pandas as pd


DATA_DIR = Path("data/raw")


def inspect_file(file_path):
    print(f"\n{'=' * 60}")
    print(f"FILE: {file_path.name}")
    print(f"{'=' * 60}")

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path, low_memory=False)
    elif file_path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type")
        return

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nFirst 5 rows:")
    print(df.head())


def main():
    for file_path in DATA_DIR.rglob("*"):
        if file_path.is_file():
            inspect_file(file_path)


if __name__ == "__main__":
    main()