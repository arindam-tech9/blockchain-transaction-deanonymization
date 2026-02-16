import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
X_train = pd.read_csv("data/processed/X_train_smote.csv")
y_train = pd.read_csv("data/processed/y_train_smote.csv").squeeze()

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# -------------------------------------------------
# Re-encode labels to 0 and 1
# -------------------------------------------------
# If labels are 1 and 2
if set(y_train.unique()) == {1, 2}:
    y_train = y_train - 1
    y_test = y_test - 1

# If labels are strings
if y_train.dtype == "object":
    y_train = y_train.map({"licit": 0, "illicit": 1})
    y_test = y_test.map({"licit": 0, "illicit": 1})

# Ensure integer type
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("Unique labels after encoding:", sorted(y_train.unique()))

# -------------------------------------------------
# Model
# -------------------------------------------------
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

print("XGBoost Results")
print("----------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
