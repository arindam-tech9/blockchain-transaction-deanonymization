import pandas as pd

features = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv")
classes = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")

print("Feature columns (first 5):")
print(features.columns.tolist()[:5])

print("\nClass columns:")
print(classes.columns.tolist())

print("\nFeatures head:")
print(features.head(3))

print("\nClasses head:")
print(classes.head(3))
