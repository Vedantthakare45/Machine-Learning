# ============================================================
# HOUSE PRICE PREDICTION
# Baseline vs Scaled Pipeline
# ============================================================

# 1. Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. Load California Housing dataset
# ============================================================

data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(X.head())


# ============================================================
# 3. Basic inspection
# ============================================================

print("\nDataset shape:")
print(X.shape)

print("\nColumn names:")
print(X.columns.tolist())

print("\nMissing values:")
print(X.isnull().sum())


# ============================================================
# 4. Split data into training and testing
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)


# ============================================================
# 5. BASELINE MODEL
#    Linear Regression WITHOUT scaling
# ============================================================

baseline_model = LinearRegression()

# Train baseline model
baseline_model.fit(X_train, y_train)

print("\nBaseline model trained successfully!")


# ============================================================
# 6. Baseline predictions
# ============================================================

baseline_pred = baseline_model.predict(X_test)


# ============================================================
# 7. Baseline evaluation
# ============================================================

baseline_mae = mean_absolute_error(
    y_test,
    baseline_pred
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_pred
)

baseline_rmse = np.sqrt(baseline_mse)

baseline_r2 = r2_score(
    y_test,
    baseline_pred
)


print("\n==============================")
print("BASELINE MODEL EVALUATION")
print("==============================")

print("MAE :", baseline_mae)
print("MSE :", baseline_mse)
print("RMSE:", baseline_rmse)
print("R2  :", baseline_r2)


# ============================================================
# 8. IMPROVED PIPELINE
#    StandardScaler + Linear Regression
# ============================================================

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])


# ============================================================
# 9. Train the scaled pipeline
# ============================================================

model.fit(X_train, y_train)

print("\nScaled pipeline trained successfully!")


# ============================================================
# 10. Make predictions
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 11. Evaluate scaled model
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("SCALED MODEL EVALUATION")
print("==============================")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# ============================================================
# 12. Compare Baseline vs Scaled Model
# ============================================================

print("\n==============================")
print("R2 COMPARISON")
print("==============================")

print("Baseline R2 :", baseline_r2)
print("Scaled R2   :", r2)

print("R2 Difference:", r2 - baseline_r2)


# ============================================================
# 13. Final conclusion
# ============================================================

if r2 > baseline_r2:
    print("\nResult: Scaling improved the R2 score.")
elif r2 < baseline_r2:
    print("\nResult: Scaling decreased the R2 score.")
else:
    print("\nResult: Scaling produced the same R2 score.")


# ============================================================
# 14. Save the improved model
# ============================================================

joblib.dump(
    model,
    "house_price_model_scaled.pkl"
)

print("\nScaled model saved successfully!")