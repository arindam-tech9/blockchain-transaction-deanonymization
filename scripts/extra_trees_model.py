import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
X_train = pd.read_csv("data/processed/X_train_smote.csv")
y_train = pd.read_csv("data/processed/y_train_smote.csv").squeeze()

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# Model
et = ExtraTreesClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

et.fit(X_train, y_train)

y_pred = et.predict(X_test)

print("Extra Trees Results")
print("--------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
