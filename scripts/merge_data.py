import pandas as pd

blockchair = pd.read_csv("data/raw/blockchair_day.csv")
wallet = pd.read_csv("data/raw/walletexplorer.csv")

merged = blockchair.merge(
    wallet,
    left_on="hash",
    right_on="transaction",
    how="inner"
)

merged.to_csv("data/processed/final_dataset.csv", index=False)

print("Final labeled dataset created.")
print("Samples:", len(merged))
