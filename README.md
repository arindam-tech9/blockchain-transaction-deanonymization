# Blockchain Transaction Deanonymization using Ensemble Learning

Hi 
This project focuses on detecting **illicit Bitcoin transactions** using ensemble machine learning models.

The main goal is simple:

> Given transaction-level blockchain data, can we automatically classify transactions as **licit** or **illicit**?

To do this, I implemented a complete machine learning pipeline — from data preprocessing to model comparison and tuning.

---

#  Why This Project?

Blockchain transactions are pseudonymous, not truly anonymous.
Detecting suspicious or illicit activity is important for:

* Fraud detection
* Financial crime monitoring
* AML (Anti-Money Laundering) systems
* Regulatory compliance

This project explores how ensemble ML methods perform on this problem.

---

#  Dataset Used

I used the **Elliptic Bitcoin Dataset**, which is a well-known benchmark dataset in blockchain research.

It contains:

* 166 numerical transaction features
* Transaction IDs
* Class labels:

  * `licit`
  * `illicit`
  * `unknown`

Before training:

* I removed the `unknown` class to make it a supervised learning problem.

---

#  Complete Project Flow (Step by Step)

Here’s exactly how the project works.

---

##  Data Cleaning & Preparation

* Loaded transaction features (the file has no headers, so I manually assigned them).
* Merged transaction features with class labels using `txId`.
* Removed transactions labeled as `unknown`.
* Saved a clean dataset for modeling.

---

##  Feature–Label Separation

* Removed `txId` to avoid data leakage.
* Split the dataset into:

  * Feature matrix `X`
  * Target variable `y`

---

##  Train–Test Split

* Used a **60:40 split**
* Applied **stratification** to preserve class distribution

This ensures fair evaluation.

---

##  Feature Scaling

* Applied **Min–Max normalization**
* Fitted the scaler on training data only
* Transformed both training and test sets

This avoids data leakage and keeps the evaluation clean.

---

##  Handling Class Imbalance (Very Important)

The dataset is highly imbalanced:

* Licit transactions >> Illicit transactions

To fix this, I used **SMOTE (Synthetic Minority Oversampling Technique)**:

* Applied only on the training set
* Generated synthetic minority samples
* Balanced the classes

The test set remained untouched.

---

##  Models Implemented

I trained and compared the following ensemble models:

* 🌳 Random Forest
* 🌲 Extra Trees
* 🧺 Bagging
* ⚡ AdaBoost
* 🚀 XGBoost

Each model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Hyperparameter Tuning

After comparing models, I selected the best-performing one and applied **GridSearchCV** to tune:

* Number of estimators
* Tree depth
* Learning rate

This improved minority-class (illicit) detection performance.

---

#  Final Outcome

* Ensemble models performed significantly better than basic classifiers.
* SMOTE improved illicit recall noticeably.
* XGBoost (in most cases) achieved the highest F1-score for illicit transactions.

* | Model    | Accuracy | Illicit Precision | Illicit Recall | Illicit F1 |
| -------- | -------- | ----------------- | -------------- | ---------- |
| RF       | 0.96     | 0.89              | 0.85           | 0.87       |
| ET       | 0.97     | 0.91              | 0.87           | 0.89       |
| Bagging  | 0.95     | 0.86              | 0.83           | 0.85       |
| AdaBoost | 0.93     | 0.82              | 0.78           | 0.80       |
| XGBoost  | 0.98     | 0.93              | 0.90           | 0.91       |


The project demonstrates how careful preprocessing + ensemble learning can effectively detect suspicious blockchain activity.

---

#  Limitations

This project focuses only on transaction-level tabular features.

It does not include:

* Temporal modeling
* Graph Neural Networks
* Wallet clustering
* Real-time blockchain streaming

These are great directions for future work.

---

#  Future Improvements

Possible extensions:

* Graph-based models (GNN)
* Time-aware sequence modeling
* Wallet-level deanonymization
* Real-time fraud detection pipeline

---

#  Tech Stack

* Python
* Pandas
* Scikit-learn
* Imbalanced-learn
* XGBoost

---

# How to Run

Clone the repository:

```
git clone <your_repo_url>
cd blockchain-transaction-deanonymization
```

Install dependencies:

```
pip install -r requirements.txt
```

Run scripts in order:

```
prepare_elliptic.py
split_features_labels.py
train_test_split.py
scale_features.py
apply_smote.py
random_forest_model.py
extra_trees_model.py
bagging_model.py
adaboost_model.py
xgboost_model.py
xgboost_tuning.py
```

---

#  Author

Arindam Sarkar
Computer Science & Data Processing
