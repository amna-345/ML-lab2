"""
Activity 2: Predict whether a customer will churn based on behavior/usage
Dataset: Telco Customer Churn (data/telco_churn.csv)
Model: Logistic Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("data/telco_churn.csv")
print("First 5 rows:\n", df.head())

# TotalCharges has some blank strings in the raw data -> convert to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Drop identifier column, not a feature
df.drop("customerID", axis=1, inplace=True)

# ---------------------------------------------------------
# 2. Encode categorical variables
# ---------------------------------------------------------
target_le = LabelEncoder()
df["Churn"] = target_le.fit_transform(df["Churn"])  # Yes=1, No=0

categorical_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ---------------------------------------------------------
# 3. Select features & target
#    (as per the lab: tenure, MonthlyCharges, Contract, InternetService
#     -- plus a few more useful ones so the model isn't too weak)
# ---------------------------------------------------------
feature_cols = ["tenure", "MonthlyCharges", "Contract", "InternetService",
                 "PaperlessBilling", "PaymentMethod", "TotalCharges"]

X = df[feature_cols]
y = df["Churn"]

# ---------------------------------------------------------
# 4. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------
# 5. Scale features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------
# 6. Train Logistic Regression model
# ---------------------------------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 7. Predict & Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))

# ---------------------------------------------------------
# 8. Confusion Matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Telco Churn")
plt.tight_layout()
plt.savefig("outputs/activity2_confusion_matrix.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 9. ROC Curve
# ---------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Telco Churn")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/activity2_roc_curve.png", dpi=150)
plt.show()
