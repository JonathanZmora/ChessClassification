import pandas as pd
import sys


if __name__ == "__main__":
    """ script to check if a column of a csv file is all unique values """
    argv: list[str] = sys.argv
    argc: int = len(argv)

    if argc != 3:
        print(f"Usage: python3 {argv[0]} <csv_file_path> <column_name>", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(argv[1])

    col = argv[2]

    # check if are there any duplicates
    has_dupes = df[col].duplicated().any()
    print("Has duplicates?", has_dupes)

    if not has_dupes:
        sys.exit(0)

    # show all occurrences of the duplicated values
    dupe_rows = df[df[col].duplicated(keep=False)].sort_values(col)
    print(dupe_rows[[col]])

    # count how many times each duplicated value appears
    counts = df[col].value_counts()
    print(counts[counts > 1])
