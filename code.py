
# ============================================================
# DAY 12: HOUSE PRICE PREDICTION USING RANDOM FOREST REGRESSOR
# Dataset: California Housing
# Models: Linear Regression, Ridge, Decision Tree, Random Forest
# ============================================================

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_data():
    print("\nLoading California Housing dataset...")

    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target

    print("Dataset loaded successfully.")
    print("Number of rows:", X.shape[0])
    print("Number of features:", X.shape[1])
    print("Features:", list(X.columns))

    return X, y


# ============================================================
# 3. INSPECT DATA
# ============================================================

def inspect_data(X, y):
    print("\n========== DATASET INFORMATION ==========")

    print("\nFirst five rows:")
    print(X.head())

    print("\nDataset information:")
    X.info()

    print("\nMissing values:")
    print(X.isnull().sum())

    print("\nTarget statistics:")
    print(y.describe())


# ============================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# ============================================================

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print("\n========== DATA SPLITTING ==========")
    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


# ============================================================
# 5. COMMON MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print(f"\n========== {model_name} RESULTS ==========")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 6. TRAIN BASELINE LINEAR REGRESSION
# ============================================================

def train_linear_regression(X_train, y_train):
    print("\nTraining Linear Regression...")

    linear_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    linear_model.fit(X_train, y_train)

    print("Linear Regression training completed.")

    return linear_model


# ============================================================
# 7. TRAIN RIDGE REGRESSION WITH GRID SEARCH
# ============================================================

def train_ridge_regression(X_train, y_train):
    print("\nTraining Ridge Regression with GridSearchCV...")

    ridge_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ])

    param_grid = {
        "model__alpha": [0.01, 0.1, 1, 10, 100]
    }

    grid_search = GridSearchCV(
        estimator=ridge_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Ridge parameters:", grid_search.best_params_)
    print("Best cross-validation R²:", round(
        grid_search.best_score_, 4
    ))

    return grid_search.best_estimator_


# ============================================================
# 8. TRAIN DECISION TREE REGRESSION
# ============================================================

def train_decision_tree(X_train, y_train):
    print("\nTraining Decision Tree Regressor...")

    tree_pipeline = Pipeline([
        ("model", DecisionTreeRegressor(random_state=42))
    ])

    param_grid = {
        "model__max_depth": [3, 5, 10, 15, 20, None],
        "model__min_samples_leaf": [1, 2, 5]
    }

    grid_search = GridSearchCV(
        estimator=tree_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Decision Tree parameters:", grid_search.best_params_)
    print("Best cross-validation R²:", round(
        grid_search.best_score_, 4
    ))

    return grid_search.best_estimator_


# ============================================================
# 9. DAY 12: RANDOM FOREST PIPELINE
# ============================================================

def train_random_forest(X_train, y_train):
    print("\n========== RANDOM FOREST TRAINING ==========")

    # Pipeline integrates Random Forest into the ML workflow.
    # StandardScaler is retained for consistency with earlier models.
    # Random Forest itself does not require feature scaling.

    random_forest_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Fit the pipeline on training data
    random_forest_pipeline.fit(X_train, y_train)

    print("Random Forest training completed.")
    print("Number of trees:", 100)

    return random_forest_pipeline


# ============================================================
# 10. EXTRACT AND SAVE RANDOM FOREST FEATURE IMPORTANCES
# ============================================================

def show_feature_importance(random_forest_model, X_train):
    print("\n========== FEATURE IMPORTANCE ==========")

    # Retrieve the RandomForestRegressor from the pipeline
    fitted_forest = random_forest_model.named_steps["model"]

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": fitted_forest.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).reset_index(drop=True)

    print(importance_df.to_string(index=False))

    # Save importance values to CSV
    importance_df.to_csv(
        "random_forest_feature_importance.csv",
        index=False
    )

    # Plot feature importance
    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(
        "random_forest_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()

    print("\nFeature importance CSV and graph saved.")

    return importance_df


# ============================================================
# 11. VISUALIZE RANDOM FOREST PREDICTIONS
# ============================================================

def visualize_random_forest(random_forest_model, X_test, y_test):
    predictions = random_forest_model.predict(X_test)

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))

    plt.scatter(
        y_test,
        predictions,
        alpha=0.4
    )

    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())

    plt.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    plt.xlabel("Actual House Values")
    plt.ylabel("Predicted House Values")
    plt.title("Random Forest: Actual vs Predicted")
    plt.tight_layout()

    plt.savefig(
        "random_forest_actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # Residual plot
    residuals = y_test - predictions

    plt.figure(figsize=(8, 6))

    plt.scatter(
        predictions,
        residuals,
        alpha=0.4
    )

    plt.axhline(y=0, linestyle="--")

    plt.xlabel("Predicted House Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Random Forest Residual Plot")
    plt.tight_layout()

    plt.savefig(
        "random_forest_residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    # Residual distribution
    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=30
    )

    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title("Random Forest Residual Distribution")
    plt.tight_layout()

    plt.savefig(
        "random_forest_residual_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()


# ============================================================
# 12. COMPARE ALL FOUR MODELS
# ============================================================

def compare_models(
    linear_model,
    ridge_model,
    tree_model,
    random_forest_model,
    X_test,
    y_test
):
    print("\n========== MODEL COMPARISON ==========")

    results = []

    results.append(
        evaluate_model(
            linear_model,
            X_test,
            y_test,
            "Linear Regression"
        )
    )

    results.append(
        evaluate_model(
            ridge_model,
            X_test,
            y_test,
            "Ridge Regression"
        )
    )

    results.append(
        evaluate_model(
            tree_model,
            X_test,
            y_test,
            "Decision Tree"
        )
    )

    results.append(
        evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "Random Forest"
        )
    )

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="R2",
        ascending=False
    ).reset_index(drop=True)

    print("\nFinal comparison table:")
    print(comparison_df.round(4).to_string(index=False))

    # Save model comparison
    comparison_df.to_csv(
        "model_comparison.csv",
        index=False
    )

    print("\nModel comparison saved to model_comparison.csv")

    # Display R² comparison
    plt.figure(figsize=(10, 6))

    plt.bar(
        comparison_df["Model"],
        comparison_df["R2"]
    )

    plt.xlabel("Model")
    plt.ylabel("Test R² Score")
    plt.title("Comparison of Regression Models")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        "model_r2_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    best_model_name = comparison_df.iloc[0]["Model"]

    print("\nBest-performing model by test R²:", best_model_name)
    print("Higher R² generally indicates a better fit.")

    return comparison_df


# ============================================================
# 13. SAVE TRAINED MODELS
# ============================================================

def save_models(
    ridge_model,
    tree_model,
    random_forest_model
):
    print("\n========== SAVING MODELS ==========")

    joblib.dump(
        ridge_model,
        "house_price_ridge_best_model.pkl"
    )

    joblib.dump(
        tree_model,
        "house_price_decision_tree_model.pkl"
    )

    # Save the complete fitted Random Forest pipeline
    joblib.dump(
        random_forest_model,
        "house_price_random_forest_model.pkl"
    )

    print("Ridge model saved.")
    print("Decision Tree model saved.")
    print("Final Random Forest model saved.")
    print("Filename: house_price_random_forest_model.pkl")


# ============================================================
# 14. MAIN PROGRAM
# ============================================================

def main():
    # Step 1: Load data
    X, y = load_data()

    # Step 2: Inspect dataset
    inspect_data(X, y)

    # Step 3: Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4: Train earlier models
    linear_model = train_linear_regression(
        X_train,
        y_train
    )

    ridge_model = train_ridge_regression(
        X_train,
        y_train
    )

    tree_model = train_decision_tree(
        X_train,
        y_train
    )

    # Step 5: Train Random Forest (Day 12)
    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    # Step 6: Evaluate all models and compare
    comparison_df = compare_models(
        linear_model,
        ridge_model,
        tree_model,
        random_forest_model,
        X_test,
        y_test
    )

    # Step 7: Feature importance
    importance_df = show_feature_importance(
        random_forest_model,
        X_train
    )

    # Step 8: Visualize Random Forest predictions
    visualize_random_forest(
        random_forest_model,
        X_test,
        y_test
    )

    # Step 9: Persist trained models
    save_models(
        ridge_model,
        tree_model,
        random_forest_model
    )

    # Step 10: Final summary
    print("\n========== DAY 12 COMPLETED ==========")
    print("Random Forest pipeline trained and evaluated.")
    print("Feature importance analysis completed.")
    print("All four models compared.")
    print("Final Random Forest model persisted successfully.")

    print("\nTop three important features:")
    print(importance_df.head(3).to_string(index=False))

    print("\nFinal model comparison:")
    print(comparison_df.round(4).to_string(index=False))


# ============================================================
# 15. RUN THE PROGRAM
# ============================================================

if __name__ == "__main__":
    main()