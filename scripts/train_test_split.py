import pandas as pd
from sklearn.model_selection import train_test_split

# Load features and labels
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv").squeeze()

# Train–test split (60:40)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42,
    stratify=y
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

# Save splits
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("\nSaved train–test split files")
