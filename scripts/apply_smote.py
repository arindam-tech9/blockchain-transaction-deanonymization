import pandas as pd
from imblearn.over_sampling import SMOTE

# -------------------------------
# Load scaled training data
# -------------------------------
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()

print("Before SMOTE class distribution:")
print(y_train.value_counts())

# -------------------------------
# Apply SMOTE
# -------------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts())

# -------------------------------
# Save balanced training data
# -------------------------------
pd.DataFrame(X_train_smote, columns=X_train.columns)\
  .to_csv("data/processed/X_train_smote.csv", index=False)

pd.Series(y_train_smote)\
  .to_csv("data/processed/y_train_smote.csv", index=False)

print("\nSaved:")
print(" - X_train_smote.csv")
print(" - y_train_smote.csv")
