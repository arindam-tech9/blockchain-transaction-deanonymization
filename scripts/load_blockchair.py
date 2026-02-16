import pandas as pd

df = pd.read_csv(
    "data/raw/transactions_20251216.csv",
    sep="\t",          # TAB separator (confirmed)
    engine="python"    # robust parser
)

print("Shape:", df.shape)
print("\nColumns (first 10):")
print(df.columns.tolist()[:10])
print("\nFirst 3 rows:")
print(df.head(3))
