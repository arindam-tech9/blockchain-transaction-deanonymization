import pandas as pd

# -------------------------------
# STEP 1: Load Elliptic features
# IMPORTANT: features file has NO header
# -------------------------------
features = pd.read_csv(
    "data/raw/elliptic/elliptic_txs_features.csv",
    header=None
)

# Manually assign column names
# 1 txId + 1 timestep + remaining feature columns
features.columns = (
    ["txId", "timestep"] +
    [f"feature_{i}" for i in range(1, features.shape[1] - 1)]
)

print("Features loaded correctly")
print("Feature columns (first 5):", features.columns[:5].tolist())
print("Features shape:", features.shape)

# -------------------------------
# STEP 2: Load class labels
# -------------------------------
classes = pd.read_csv(
    "data/raw/elliptic/elliptic_txs_classes.csv"
)

print("\nClasses loaded correctly")
print("Class columns:", classes.columns.tolist())
print("Classes shape:", classes.shape)

# -------------------------------
# STEP 3: Merge features + labels
# -------------------------------
df = features.merge(
    classes,
    on="txId",
    how="inner"
)

print("\nAfter merge:")
print("Merged shape:", df.shape)

# -------------------------------
# STEP 4: Remove 'unknown' class
# -------------------------------
df = df[df["class"] != "unknown"]

print("\nFinal class distribution (after removing unknown):")
print(df["class"].value_counts())

# -------------------------------
# STEP 5: Save clean dataset
# -------------------------------
df.to_csv(
    "data/processed/elliptic_clean.csv",
    index=False
)

print("\nSaved: data/processed/elliptic_clean.csv")
