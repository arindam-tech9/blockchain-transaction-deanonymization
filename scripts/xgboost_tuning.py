import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed/X_train_smote.csv")
y_train = pd.read_csv("data/processed/y_train_smote.csv").squeeze()

# Ensure labels 0/1
if set(y_train.unique()) == {1, 2}:
    y_train = y_train - 1

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1]
}

xgb = XGBClassifier(eval_metric='logloss')

grid = GridSearchCV(
    xgb,
    param_grid,
    cv=3,
    scoring="f1",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best F1 Score:", grid.best_score_)
