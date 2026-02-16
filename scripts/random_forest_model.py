import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# Load data
# -------------------------------
X_train = pd.read_csv("data/processed/X_train_smote.csv")
y_train = pd.read_csv("data/processed/y_train_smote.csv").squeeze()

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# -------------------------------
# Train Random Forest
# -------------------------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------
y_pred = rf.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Results")
print("---------------------")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
