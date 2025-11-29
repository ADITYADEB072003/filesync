# train_closing_balance.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from io import StringIO
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
    accuracy_score,
    classification_report
)

df=pd.read_csv("extracted_statement.csv")
# robustly convert Withdrawal to numeric, then fill NaNs with the column mean
col = 'Withdrawal'
if col not in df.columns:
    raise KeyError(f"Column '{col}' not found in CSV. Available columns: {df.columns.tolist()}")

# remove common formatting, convert to numeric
df[col] = (
    df[col].astype(str)
          .str.replace(',', '', regex=False)   # remove thousands separators
          .str.replace('₹', '', regex=False)   # remove rupee sign (add more symbols if needed)
          .str.replace('Rs', '', case=False, regex=False)
          .str.strip()
)

# Convert to numeric (invalid parsing -> NaN), compute mean ignoring NaNs
df[col] = pd.to_numeric(df[col], errors='coerce')
mean_val = df[col].mean()

# If mean_val is NaN (all values missing/invalid), set to 0 (or choose another default)
if pd.isna(mean_val):
    mean_val = 0.0
    print(f"Warning: no numeric values found in '{col}'; filling with {mean_val}")

df[col] = df[col].fillna(mean_val)

print(f"Filled NaNs in '{col}' with mean = {mean_val}")
tcol = 'Closing Balance'
if tcol in df.columns:
    df[tcol] = (
        df[tcol].astype(str)
                .str.replace(',', '', regex=False)
                .str.replace('₹', '', regex=False)
                .str.replace('Rs', '', case=False, regex=False)
                .str.strip()
    )
    df[tcol] = pd.to_numeric(df[tcol], errors='coerce')
    df[tcol] = df[tcol].fillna(df[tcol].mean())   # or df.dropna(subset=[tcol]) if you prefer
# print(df.head())
# --- ensure X and y are correct shapes ---
X = df[["Withdrawal"]].values            # shape (n_samples, 1)
y = df["Closing Balance"].values         # shape (n_samples,)

# Use test_size (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred_test = model.predict(X_test)
print("Y_Train",y_pred_test)
# Predict a single new value (example withdrawal = 1234)
# Predict a single new value (example withdrawal = 1234)
single = np.array([[1234.0]])            # must be 2D
single_pred = model.predict(single)
print("Predicted closing balance for withdrawal=1234:", single_pred[0])

# Regression metrics on test set (recommended)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"Test MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

# Tolerance-based "accuracy" (percent within ±5%, ±10%, ±15%)
tolerances = [0.05, 0.10, 0.15]
for tol in tolerances:
    # safe denominator, avoid division by zero
    denom = np.where(np.abs(y_test) < 1e-8, 1e-8, np.abs(y_test))
    within = np.abs(y_pred_test - y_test) <= tol * denom
    pct = np.mean(within) * 100
    print(f"Within ±{int(tol*100)}%: {pct:.2f}%")

# Scatter actual vs predicted (test set)
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, label='Actual (test)', alpha=0.6)
plt.scatter(X_test, y_pred_test, label='Predicted (test)', alpha=0.6)
# Optionally plot a regression line built from sorted X_test for visual clarity
order = np.argsort(X_test.ravel())
plt.plot(X_test.ravel()[order], y_pred_test[order], color='red', linewidth=2, label='Prediction line')
plt.xlabel("Withdrawal")
plt.ylabel("Closing Balance")
plt.title("Actual vs Predicted (Test set)")
plt.legend()
plt.grid(True)
plt.show()