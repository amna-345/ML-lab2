"""
Activity 3: Predict used car prices
Dataset: Car Price Prediction (data/car_price.csv)
Models: Linear Regression + Polynomial Regression (degree 2, 3, 4)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv("data/car_price.csv")
print("Columns available:\n", df.columns.tolist())
print(df.head())

# ---------------------------------------------------------
# 2. Choose features & target
#    Using horsepower as the primary feature (like the lab's single-feature
#    polynomial example), and price as target.
# ---------------------------------------------------------
X = df[["horsepower"]].values
y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Linear Regression (baseline)
# ---------------------------------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)
print(f"\nLinear Regression -> RMSE: {rmse_lin:.2f}, R2: {r2_lin:.4f}")

# ---------------------------------------------------------
# 4. Polynomial Regression for degrees 2, 3, 4
# ---------------------------------------------------------
degrees = [2, 3, 4]
results = {"Linear (deg=1)": (rmse_lin, r2_lin)}

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", alpha=0.5, label="Actual Data")

# sort X for smooth line plotting
X_sorted = np.sort(X, axis=0)

# plot linear fit
plt.plot(X_sorted, lin_model.predict(X_sorted), color="black",
         linewidth=2, label="Linear Fit (deg=1)")

colors = ["red", "green", "purple"]
for deg, color in zip(degrees, colors):
    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    y_pred_poly = poly_model.predict(X_test_poly)
    rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
    r2_poly = r2_score(y_test, y_pred_poly)
    results[f"Polynomial (deg={deg})"] = (rmse_poly, r2_poly)

    # smooth curve for plotting
    X_sorted_poly = poly.transform(X_sorted)
    y_sorted_pred = poly_model.predict(X_sorted_poly)
    plt.plot(X_sorted, y_sorted_pred, color=color, linewidth=2,
              label=f"Polynomial Fit (deg={deg})")

plt.xlabel("Horsepower")
plt.ylabel("Price")
plt.title("Linear vs Polynomial Regression - Car Price Prediction")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/activity3_regression_curves.png", dpi=150)
plt.show()

# ---------------------------------------------------------
# 5. Compare results
# ---------------------------------------------------------
print("\nModel Comparison:")
print(f"{'Model':<20}{'RMSE':>12}{'R2 Score':>12}")
for model_name, (rmse, r2) in results.items():
    print(f"{model_name:<20}{rmse:>12.2f}{r2:>12.4f}")
