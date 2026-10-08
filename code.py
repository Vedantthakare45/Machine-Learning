# ============================================================
# HOUSE PRICE PREDICTION
# Linear Regression + Ridge Regression + Decision Tree
# GridSearchCV + Visualization + Model Comparison
# ============================================================


# ============================================================
# 1. Import Libraries
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.tree import DecisionTreeRegressor

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
# 5. Train Baseline Linear Regression
# ============================================================

def train_linear_regression(X_train, y_train):
    """Train the baseline Linear Regression model."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("linear", LinearRegression())
    ])

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nBaseline Linear Regression "
        "trained successfully!"
    )

    return model


# ============================================================
# 6. Create Ridge Pipeline
# ============================================================

def create_ridge_pipeline():
    """Create StandardScaler + Ridge Regression pipeline."""

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge())
    ])

    return model


# ============================================================
# 7. Ridge Regression + GridSearchCV
# ============================================================

def train_ridge_with_gridsearch(X_train, y_train):
    """Find the best Ridge alpha using GridSearchCV."""

    model = create_ridge_pipeline()

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
    print(
        grid_search.best_params_["ridge__alpha"]
    )

    print("\nBest Cross-Validation R2:")
    print(
        grid_search.best_score_
    )

    return grid_search.best_estimator_


# ============================================================
# 8. Evaluate Model
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
# 9. Display Evaluation Results
# ============================================================

def display_metrics(
    title,
    mae,
    mse,
    rmse,
    r2
):
    """Display model evaluation metrics."""

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    print("MAE :", mae)
    print("MSE :", mse)
    print("RMSE:", rmse)
    print("R2  :", r2)


# ============================================================
# 10. Display Best Ridge Model
# ============================================================

def display_best_model(model):
    """Display the best Ridge model parameters."""

    best_alpha = model.named_steps["ridge"].alpha

    print("\n" + "=" * 50)
    print("BEST RIDGE MODEL")
    print("=" * 50)

    print("Best Alpha:", best_alpha)


# ============================================================
# 11. Visualization and Residual Analysis
# ============================================================

def visualize_model(model, X_test, y_test):
    """Create and save prediction and residual plots."""

    # Generate predictions
    y_pred = model.predict(X_test)

    print("\nPredictions generated successfully!")

    # --------------------------------------------------------
    # Actual vs Predicted Plot
    # --------------------------------------------------------

    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        y_pred,
        alpha=0.6
    )

    # Perfect prediction reference line
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
    # Calculate Residuals
    # --------------------------------------------------------

    residuals = y_test - y_pred

    print("\nResiduals calculated successfully!")

    # --------------------------------------------------------
    # Residual Plot
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
    # Residual Distribution
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

    print("\nVisualization plots saved successfully!")

    print("1. actual_vs_predicted.png")
    print("2. residual_plot.png")
    print("3. residual_distribution.png")


# ============================================================
# 12. Train Decision Tree with Different max_depth Values
# ============================================================

def train_decision_tree(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train Decision Tree models with different max_depth
    values and compare their R2 scores.
    """

    # Depth values to test
    depths = [
        1,
        2,
        3,
        5,
        10,
        15,
        20
    ]

    results = []

    best_model = None
    best_depth = None
    best_r2 = -np.inf

    print("\n" + "=" * 50)
    print("DECISION TREE REGRESSION")
    print("=" * 50)

    # Test each depth
    for depth in depths:

        model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        # Train the model
        model.fit(
            X_train,
            y_train
        )

        # Make predictions
        predictions = model.predict(
            X_test
        )

        # Calculate R2
        r2 = r2_score(
            y_test,
            predictions
        )

        # Store result
        results.append({
            "max_depth": depth,
            "R2": r2
        })

        print(
            f"max_depth = {depth:<3} "
            f"R2 = {r2:.4f}"
        )

        # Check whether this is the best model
        if r2 > best_r2:

            best_r2 = r2
            best_depth = depth
            best_model = model

    # Create DataFrame
    results_df = pd.DataFrame(
        results
    )

    print("\nDecision Tree Depth Results:")
    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nBest Decision Tree max_depth:")
    print(best_depth)

    print("\nBest Decision Tree R2:")
    print(best_r2)

    # Save depth experiment results
    results_df.to_csv(
        "decision_tree_depth_results.csv",
        index=False
    )

    print(
        "\nDepth results saved as:"
        " decision_tree_depth_results.csv"
    )

    return (
        best_model,
        best_depth,
        results_df
    )


# ============================================================
# 13. Evaluate Best Decision Tree
# ============================================================

def evaluate_decision_tree(
    model,
    X_test,
    y_test
):
    """Evaluate the best Decision Tree model."""

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    display_metrics(
        "BEST DECISION TREE EVALUATION",
        *metrics
    )

    return metrics


# ============================================================
# 14. Compare Models
# ============================================================

def compare_models(
    linear_metrics,
    ridge_metrics,
    tree_metrics
):
    """
    Create a comparison table for Linear Regression,
    Ridge Regression and Decision Tree Regression.
    """

    linear_mae, linear_mse, linear_rmse, linear_r2 = (
        linear_metrics
    )

    ridge_mae, ridge_mse, ridge_rmse, ridge_r2 = (
        ridge_metrics
    )

    tree_mae, tree_mse, tree_rmse, tree_r2 = (
        tree_metrics
    )

    # Create comparison DataFrame
    comparison = pd.DataFrame({

        "Model": [
            "Linear Regression",
            "Ridge Regression",
            "Decision Tree"
        ],

        "MAE": [
            linear_mae,
            ridge_mae,
            tree_mae
        ],

        "MSE": [
            linear_mse,
            ridge_mse,
            tree_mse
        ],

        "RMSE": [
            linear_rmse,
            ridge_rmse,
            tree_rmse
        ],

        "R2": [
            linear_r2,
            ridge_r2,
            tree_r2
        ]
    })

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison.to_string(
            index=False
        )
    )

    # Save comparison table
    comparison.to_csv(
        "model_comparison.csv",
        index=False
    )

    print(
        "\nModel comparison saved as:"
        " model_comparison.csv"
    )

    return comparison


# ============================================================
# 15. Display Final Insight
# ============================================================

def display_final_insight(comparison):
    """Display which model performed best based on R2."""

    best_index = comparison["R2"].idxmax()

    best_model_name = comparison.loc[
        best_index,
        "Model"
    ]

    best_r2 = comparison.loc[
        best_index,
        "R2"
    ]

    print("\n" + "=" * 50)
    print("FINAL MODEL INSIGHT")
    print("=" * 50)

    print(
        "Best model based on test R2:",
        best_model_name
    )

    print(
        "Best test R2:",
        best_r2
    )

    print(
        "\nA higher R2 indicates better "
        "explanation of variation in house prices."
    )


# ============================================================
# 16. Save Best Ridge Model
# ============================================================

def save_ridge_model(model):
    """Save the best Ridge model."""

    model_filename = (
        "house_price_ridge_best_model.pkl"
    )

    joblib.dump(
        model,
        model_filename
    )

    print(
        "\nBest Ridge model saved successfully!"
    )

    print(
        "Saved file:",
        model_filename
    )


# ============================================================
# 17. Save Best Decision Tree Model
# ============================================================

def save_tree_model(model):
    """Save the best Decision Tree model."""

    model_filename = (
        "house_price_decision_tree_model.pkl"
    )

    joblib.dump(
        model,
        model_filename
    )

    print(
        "\nBest Decision Tree model saved successfully!"
    )

    print(
        "Saved file:",
        model_filename
    )


# ============================================================
# 18. Main Program
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

    # ========================================================
    # BASELINE LINEAR REGRESSION
    # ========================================================

    linear_model = train_linear_regression(
        X_train,
        y_train
    )

    linear_metrics = evaluate_model(
        linear_model,
        X_test,
        y_test
    )

    display_metrics(
        "BASELINE LINEAR REGRESSION",
        *linear_metrics
    )

    # ========================================================
    # RIDGE REGRESSION + GRIDSEARCHCV
    # ========================================================

    best_ridge_model = train_ridge_with_gridsearch(
        X_train,
        y_train
    )

    # Display best Ridge model
    display_best_model(
        best_ridge_model
    )

    # Evaluate Ridge
    ridge_metrics = evaluate_model(
        best_ridge_model,
        X_test,
        y_test
    )

    display_metrics(
        "BEST RIDGE MODEL EVALUATION",
        *ridge_metrics
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    visualize_model(
        best_ridge_model,
        X_test,
        y_test
    )

    # ========================================================
    # DECISION TREE REGRESSION
    # ========================================================

    (
        best_tree_model,
        best_tree_depth,
        tree_depth_results
    ) = train_decision_tree(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ========================================================
    # EVALUATE BEST DECISION TREE
    # ========================================================

    tree_metrics = evaluate_decision_tree(
        best_tree_model,
        X_test,
        y_test
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    comparison_table = compare_models(
        linear_metrics,
        ridge_metrics,
        tree_metrics
    )

    # ========================================================
    # FINAL INSIGHT
    # ========================================================

    display_final_insight(
        comparison_table
    )

    # ========================================================
    # SAVE MODELS
    # ========================================================

    save_ridge_model(
        best_ridge_model
    )

    save_tree_model(
        best_tree_model
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n" + "=" * 70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    print("\nBest Decision Tree Depth:")
    print(best_tree_depth)

    print("\nGenerated Files:")

    print("1. actual_vs_predicted.png")
    print("2. residual_plot.png")
    print("3. residual_distribution.png")
    print("4. decision_tree_depth_results.csv")
    print("5. model_comparison.csv")
    print("6. house_price_ridge_best_model.pkl")
    print("7. house_price_decision_tree_model.pkl")


# ============================================================
# 19. Entry Point
# ============================================================

if __name__ == "__main__":
    main()