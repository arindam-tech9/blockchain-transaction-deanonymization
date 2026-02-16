
import pandas as pd

blockchair = pd.read_csv("data/raw/transactions_20251216.csv")

print("Blockchair columns:")
print(blockchair.columns.tolist())

print("\nBlockchair shape:", blockchair.shape)
print("\nFirst 5 rows:")
print(blockchair.head())