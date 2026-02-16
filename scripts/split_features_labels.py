import pandas as pd

# Load cleaned Elliptic dataset
df = pd.read_csv("data/processed/elliptic_clean.csv")

# Separate features and labels
X = df.drop(columns=["txId", "class"])
y = df["class"]

print("Feature matrix shape:", X.shape)
print("Label distribution:")
print(y.value_counts())

# Save
X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)

print("\nSaved:")
print(" - data/processed/X.csv")
print(" - data/processed/y.csv")
