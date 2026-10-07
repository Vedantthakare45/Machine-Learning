# ============================================================
# HOUSE PRICE PREDICTION
# Ridge Regression + GridSearchCV + Visualization
# ============================================================


# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
# 5. Create Ridge Pipeline
# ============================================================

def create_ridge_pipeline():
    """Create StandardScaler + Ridge Regression pipeline."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    return model


# ============================================================
# 6. GridSearchCV for Best Alpha
# ============================================================

def train_ridge_with_gridsearch(X_train, y_train):
    """Find the best Ridge alpha using GridSearchCV."""

    model = create_ridge_pipeline()

    # Alpha values to test
    param_grid = {
        "ridge__alpha": [
            0.01,
            0.1,
            1,
            10,
            100
        ]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    print("\nGridSearchCV completed successfully!")

    print("\nBest Alpha:")
    print(grid_search.best_params_["ridge__alpha"])

    print("\nBest Cross-Validation R2:")
    print(grid_search.best_score_)

    return grid_search.best_estimator_


# ============================================================
# 7. Evaluate Model
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
# 8. Display Evaluation Results
# ============================================================

def display_metrics(title, mae, mse, rmse, r2):
    """Display model evaluation metrics."""

    print("\n" + "=" * 40)
    print(title)
    print("=" * 40)

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ============================================================
# 9. Display Best Model Information
# ============================================================

def display_best_model(model):
    """Display the best Ridge model parameters."""

    best_alpha = model.named_steps["ridge"].alpha

    print("\n" + "=" * 40)
    print("BEST RIDGE MODEL")
    print("=" * 40)

    print("Best Alpha:", best_alpha)


# ============================================================
# 10. Visualization and Residual Analysis
# ============================================================

def visualize_model(model, X_test, y_test):
    """Create and save prediction and residual plots."""

    # --------------------------------------------------------
    # Generate Predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    print("\nPredictions generated successfully!")

    # --------------------------------------------------------
    # 1. Actual vs Predicted Plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    # Perfect prediction line
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted House Prices")

    plt.tight_layout()

    plt.savefig(
        "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # --------------------------------------------------------
    # 2. Calculate Residuals
    # --------------------------------------------------------

    # Residual = Actual - Predicted

    residuals = y_test - y_pred

    print("\nResiduals calculated successfully!")

    # --------------------------------------------------------
    # 3. Residual Plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    # Zero residual reference line
    plt.axhline(
        y=0,
        linestyle="--"
    )

    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    plt.tight_layout()

    plt.savefig(
        "residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # --------------------------------------------------------
    # 4. Residual Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Residuals")

    plt.tight_layout()

    plt.savefig(
        "residual_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()

    # --------------------------------------------------------
    # Display Saved Plot Information
    # --------------------------------------------------------

    print("\nVisualization plots saved successfully!")

    print("1. actual_vs_predicted.png")
    print("2. residual_plot.png")
    print("3. residual_distribution.png")


# ============================================================
# 11. Save Best Model
# ============================================================

def save_model(model):
    """Save the best trained Ridge pipeline using Joblib."""

    model_filename = "house_price_ridge_best_model.pkl"

    joblib.dump(
        model,
        model_filename
    )

    print("\nBest Ridge model saved successfully!")
    print("Saved file:", model_filename)


# ============================================================
# 12. Main Program
# ============================================================

def main():

    # --------------------------------------------------------
    # Load Dataset
    # --------------------------------------------------------

    X, y = load_data()

    # --------------------------------------------------------
    # Inspect Dataset
    # --------------------------------------------------------

    inspect_data(X)

    # --------------------------------------------------------
    # Split Dataset
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    # --------------------------------------------------------
    # Ridge Regression + GridSearchCV
    # --------------------------------------------------------

    best_model = train_ridge_with_gridsearch(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Display Best Model
    # --------------------------------------------------------

    display_best_model(best_model)

    # --------------------------------------------------------
    # Evaluate Best Model
    # --------------------------------------------------------

    mae, mse, rmse, r2 = evaluate_model(
        best_model,
        X_test,
        y_test
    )

    display_metrics(
        "BEST RIDGE MODEL EVALUATION",
        mae,
        mse,
        rmse,
        r2
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualize_model(
        best_model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save Best Model
    # --------------------------------------------------------

    save_model(best_model)


# ============================================================
# 13. Entry Point
# ============================================================

if __name__ == "__main__":
    main()