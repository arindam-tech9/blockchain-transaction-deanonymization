import pandas as pd

blockchair = pd.read_csv("data/processed/blockchair_clean.csv")
wallet = pd.read_csv("data/processed/walletexplorer_clean.csv")

merged = blockchair.merge(
    wallet,
    on="hash",
    how="inner"
)

merged.to_csv("data/processed/final_labeled_dataset.csv", index=False)

print("Merged dataset created")
print("Merged shape:", merged.shape)
print("Label distribution:")
print(merged["label"].value_counts())
