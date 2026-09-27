"""
Activity 1: Predict a person's medical insurance cost based on personal details
Dataset: Medical Cost Personal Dataset (data/medical_cost.csv)
Model: Linear Regression
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load and explore the dataset
# ---------------------------------------------------------
df = pd.read_csv("data/medical_cost.csv")
print("First 5 rows:\n", df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Features: age, bmi, children, smoker, region (+ sex, kept as it's categorical too)
# Target: charges (insurance cost)

# ---------------------------------------------------------
# 2. Preprocess - Encode categorical features
# ---------------------------------------------------------
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df["sex"] = le_sex.fit_transform(df["sex"])          # male=1, female=0
df["smoker"] = le_smoker.fit_transform(df["smoker"])  # yes=1, no=0
df["region"] = le_region.fit_transform(df["region"])  # encoded 0..3

# ---------------------------------------------------------
# 3. Split features/target
# ---------------------------------------------------------
X = df.drop("charges", axis=1)
y = df["charges"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4. Scale numerical features
# ---------------------------------------------------------
num_cols = ["age", "bmi", "children"]
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# 5. Train Linear Regression model
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Predict and Evaluate
# ---------------------------------------------------------
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nRMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# ---------------------------------------------------------
# 7. Plot Predicted vs Actual Costs
# ---------------------------------------------------------
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color="teal", alpha=0.6, label="Predictions")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red", linewidth=2, label="Ideal Fit"
)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual Medical Insurance Cost")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/activity1_predicted_vs_actual.png", dpi=150)
plt.show()
