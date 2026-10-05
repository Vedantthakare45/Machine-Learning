# House Price Prediction Using Linear Regression

## Project Description

This project builds a complete machine learning pipeline for predicting house prices using the **California Housing dataset**.

The project uses Python, Pandas, NumPy, Scikit-learn, and Joblib to perform data loading, data inspection, train-test splitting, baseline model training, feature scaling, Linear Regression, model evaluation, model comparison, and model persistence.

The project is progressively improved as part of the **SkillPilot Machine Learning learning roadmap**.

---

## Technologies and Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

---

## Dataset

The project uses the **California Housing dataset** provided by Scikit-learn.

The dataset contains numerical features related to California housing districts and a target value representing the median house value.

The input features include:

* MedInc
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

Since all input features in this dataset are numerical, **One-Hot Encoding is not required**.

---

## Machine Learning Workflow

```text
California Housing Dataset
          ↓
     Data Loading
          ↓
    Data Inspection
          ↓
    Train/Test Split
          ↓
    Baseline Model
          ↓
   Model Evaluation
          ↓
    Feature Scaling
          ↓
     StandardScaler
          ↓
    Linear Regression
          ↓
      Prediction
          ↓
     Evaluation
          ↓
Baseline vs Scaled R²
      Comparison
          ↓
   Model Persistence
          ↓
 house_price_model_scaled.pkl
```

---

## Train-Test Split

The dataset is divided into:

* **80% training data**
* **20% testing data**

The training data is used to train the models, while the testing data is used to evaluate their performance on unseen data.

The split uses:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# Baseline Model

The first model is a standard **Linear Regression** model without feature scaling.

```python
baseline_model = LinearRegression()

baseline_model.fit(
    X_train,
    y_train
)
```

The baseline model provides a reference point for comparing the scaled pipeline.

---

# Feature Scaling

The second model uses **StandardScaler** before Linear Regression.

StandardScaler transforms numerical features to a comparable scale.

The preprocessing and regression model are combined into a Scikit-learn Pipeline:

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])
```

Using a Pipeline ensures that the same preprocessing steps are applied consistently during training and prediction.

---

# Scaled Model

The scaled model consists of:

1. `StandardScaler`
2. `LinearRegression`

The pipeline is trained using the training dataset:

```python
model.fit(
    X_train,
    y_train
)
```

Predictions are then generated using:

```python
predictions = model.predict(X_test)
```

---

# Evaluation Metrics

The models are evaluated using four regression metrics.

### MAE — Mean Absolute Error

Measures the average absolute difference between the actual and predicted values.

### MSE — Mean Squared Error

Measures the average squared difference between actual and predicted values.

### RMSE — Root Mean Squared Error

RMSE is the square root of MSE and represents prediction error in the same units as the target.

### R² — R-squared

Measures how well the model explains the variation in the target variable.

The project calculates:

```text
MAE
MSE
RMSE
R²
```

for both the baseline and scaled models.

---

# Baseline vs Scaled Model Comparison

The project compares the R² scores of both models.

```python
r2_difference = scaled_r2 - baseline_r2
```

The following values are displayed:

```text
Baseline R²
Scaled Model R²
R² Difference
```

The project also determines whether scaling improved, decreased, or produced the same R² score.

```text
If Scaled R² > Baseline R²:
    Scaling improved the R² score.

If Scaled R² < Baseline R²:
    Scaling decreased the R² score.

Otherwise:
    Scaling produced the same R² score.
```

### Important Note

For Linear Regression, feature scaling may produce the same or a nearly identical R² score because Linear Regression is not generally dependent on feature scale for predictive performance.

The purpose of this comparison is to understand and demonstrate the **feature scaling and pipeline workflow**.

---

# Categorical Encoding

The California Housing dataset used in this project contains only numerical input features.

Therefore, **One-Hot Encoding is not required** for this project.

For a different housing dataset containing categorical features such as:

* Location
* Furnishing Status
* Property Type

categorical features could be processed using `OneHotEncoder` together with `ColumnTransformer`.

---

# Project Structure

```text
House-Price-Prediction/
│
├── code.py
├── house_price_model_scaled.pkl
└── README.md
```

### `code.py`

Contains the complete machine learning workflow, including:

* Dataset loading
* Data inspection
* Train-test splitting
* Baseline model
* Feature scaling
* Linear Regression
* Model evaluation
* R² comparison
* Model persistence

### `house_price_model_scaled.pkl`

Contains the trained Scikit-learn pipeline saved using Joblib.

### `README.md`

Contains the project documentation.

---

# Code Structure

The project is organized into reusable functions instead of placing all logic in the global script scope.

Main functions include:

```python
load_data()
inspect_data()
split_data()
evaluate_model()
train_baseline_model()
train_scaled_model()
display_metrics()
compare_models()
save_model()
main()
```

The program uses the standard Python entry-point pattern:

```python
if __name__ == "__main__":
    main()
```

This makes the code easier to reuse, test, and maintain.

---

# Model Persistence

After training and evaluation, the final scaled pipeline is saved using Joblib.

```python
joblib.dump(
    model,
    "house_price_model_scaled.pkl"
)
```

The saved model contains both:

* `StandardScaler`
* `LinearRegression`

Therefore, the complete preprocessing and prediction pipeline can be reused later without retraining the model.

---

# Loading the Saved Model

The saved pipeline can later be loaded using:

```python
import joblib

model = joblib.load(
    "house_price_model_scaled.pkl"
)
```

After loading, predictions can be made using new input data:

```python
prediction = model.predict(new_data)
```

---

# Project Goals

The main goals of this project are to demonstrate a complete machine learning workflow:

* Load a real-world dataset
* Inspect the dataset
* Check for missing values
* Split data into training and testing sets
* Build a baseline regression model
* Apply feature scaling
* Create a machine learning Pipeline
* Train a Linear Regression model
* Generate predictions
* Evaluate model performance
* Compare baseline and scaled R² scores
* Calculate R² difference
* Save the trained model
* Organize the code using reusable functions

---

# Day 8 — Feature Scaling and Model Comparison

As part of **Day 8 of the SkillPilot Machine Learning roadmap**, the existing House Price Prediction project was extended with:

* Feature scaling using `StandardScaler`
* Pipeline-based preprocessing
* Baseline model comparison
* Scaled model evaluation
* R² comparison
* R² difference calculation
* Model persistence using Joblib
* Function-based code organization

This improvement demonstrates how preprocessing and model comparison can be incorporated into a machine learning workflow.

---

# Conclusion

The House Price Prediction project demonstrates a complete and reusable regression workflow using **Linear Regression**.

The project compares a baseline Linear Regression model with a scaled Linear Regression pipeline and evaluates both using MAE, MSE, RMSE, and R².

The final scaled pipeline is saved using Joblib, allowing it to be loaded and reused for future predictions without retraining.

This project provides practical experience with **data preprocessing, feature scaling, pipelines, regression evaluation, model comparison, code organization, and model persistence**.
