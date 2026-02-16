import pandas as pd

df = pd.read_csv(
    "data/raw/transactions_20251216.csv",
    sep="\t",
    low_memory=False
)

print(df.head())
print(df.columns)
