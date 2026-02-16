import pandas as pd

df = pd.read_csv(
    "data/raw/transactions_20251216.csv",
    sep="\t",
    engine="python"
)

print("Shape:", df.shape)
print("Columns (first 10):", df.columns.tolist()[:10])
print(df.head())
