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
# 2. Load Dataset
# ============================================================

def load_data():
    """Load and prepare the California Housing dataset."""

    data = fetch_california_housing()

    X = pd.DataFrame(
        data.data,
        columns=data.feature_names
    )

    y = data.target

    print("Dataset loaded successfully!")

    return X, y


# ============================================================
# 3. Inspect Dataset
# ============================================================

def inspect_data(X):
    """Display basic information about the dataset."""

    print("\nFirst 5 rows:")
    print(X.head())

    print("\nDataset shape:")
    print(X.shape)

    print("\nColumn names:")
    print(X.columns.tolist())

    print("\nMissing values:")
    print(X.isnull().sum())


# ============================================================
# 4. Split Dataset
# ============================================================

def split_data(X, y):
    """Split data into training and testing sets."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\nTraining data shape:", X_train.shape)
    print("Testing data shape :", X_test.shape)

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. Evaluate Model
# ============================================================

def evaluate_model(model, X_test, y_test):
    """Calculate regression evaluation metrics."""

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    return mae, mse, rmse, r2


# ============================================================
# 6. Baseline Model
# ============================================================

def train_baseline_model(X_train, y_train):
    """Train Linear Regression without feature scaling."""

    baseline_model = LinearRegression()

    baseline_model.fit(
        X_train,
        y_train
    )

    print("\nBaseline model trained successfully!")

    return baseline_model


# ============================================================
# 7. Scaled Pipeline
# ============================================================

def train_scaled_model(X_train, y_train):
    """Train StandardScaler + Linear Regression pipeline."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regression", LinearRegression())
    ])

    model.fit(
        X_train,
        y_train
    )

    print("Scaled pipeline trained successfully!")

    return model


# ============================================================
# 8. Display Evaluation Results
# ============================================================

def display_metrics(title, mae, mse, rmse, r2):
    """Display model evaluation metrics."""

    print("\n" + "=" * 35)
    print(title)
    print("=" * 35)

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ============================================================
# 9. Compare Models
# ============================================================

def compare_models(baseline_r2, scaled_r2):
    """Compare baseline and scaled model R² scores."""

    r2_difference = scaled_r2 - baseline_r2

    print("\n" + "=" * 35)
    print("R2 COMPARISON")
    print("=" * 35)

    print("Baseline R2 :", baseline_r2)
    print("Scaled R2   :", scaled_r2)
    print("R2 Difference:", r2_difference)

    if scaled_r2 > baseline_r2:
        print("\nResult: Scaling improved the R2 score.")

    elif scaled_r2 < baseline_r2:
        print("\nResult: Scaling decreased the R2 score.")

    else:
        print("\nResult: Scaling produced the same R2 score.")

    return r2_difference


# ============================================================
# 10. Save Model
# ============================================================

def save_model(model):
    """Save the trained scaled pipeline using Joblib."""

    model_filename = "house_price_model_scaled.pkl"

    joblib.dump(
        model,
        model_filename
    )

    print("\nScaled model saved successfully!")
    print("Saved file:", model_filename)


# ============================================================
# 11. Main Program
# ============================================================

def main():

    # Load dataset
    X, y = load_data()

    # Inspect dataset
    inspect_data(X)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Baseline Model
    # --------------------------------------------------------

    baseline_model = train_baseline_model(
        X_train,
        y_train
    )

    baseline_mae, baseline_mse, baseline_rmse, baseline_r2 = (
        evaluate_model(
            baseline_model,
            X_test,
            y_test
        )
    )

    display_metrics(
        "BASELINE MODEL EVALUATION",
        baseline_mae,
        baseline_mse,
        baseline_rmse,
        baseline_r2
    )

    # --------------------------------------------------------
    # Scaled Pipeline
    # --------------------------------------------------------

    model = train_scaled_model(
        X_train,
        y_train
    )

    mae, mse, rmse, r2 = evaluate_model(
        model,
        X_test,
        y_test
    )

    display_metrics(
        "SCALED MODEL EVALUATION",
        mae,
        mse,
        rmse,
        r2
    )

    # --------------------------------------------------------
    # Compare Baseline vs Scaled Model
    # --------------------------------------------------------

    compare_models(
        baseline_r2,
        r2
    )

    # --------------------------------------------------------
    # Save Final Model
    # --------------------------------------------------------

    save_model(model)


# ============================================================
# 12. Entry Point
# ============================================================

if __name__ == "__main__":
    main()