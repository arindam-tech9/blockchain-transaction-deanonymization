import pandas as pd

df = pd.read_csv("data/processed/walletexplorer_clean.csv")

print("Shape:", df.shape)
print("Labels:")
print(df["label"].value_counts())
print("\nFirst 5 rows:")
print(df.head())
