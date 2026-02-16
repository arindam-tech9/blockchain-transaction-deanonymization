import pandas as pd

# Replace numbers below with your actual results
results = {
    "Model": [
        "Random Forest",
        "Extra Trees",
        "Bagging",
        "AdaBoost",
        "XGBoost"
    ],
    "Accuracy": [0.96, 0.97, 0.95, 0.93, 0.98],
    "Illicit Precision": [0.89, 0.91, 0.86, 0.82, 0.93],
    "Illicit Recall": [0.85, 0.87, 0.83, 0.78, 0.90],
    "Illicit F1": [0.87, 0.89, 0.85, 0.80, 0.91]
}

df = pd.DataFrame(results)

print("\nModel Comparison Table:\n")
print(df)

df.to_csv("data/processed/model_comparison.csv", index=False)

print("\nSaved: data/processed/model_comparison.csv")

