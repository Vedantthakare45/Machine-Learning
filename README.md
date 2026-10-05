# House Price Prediction Using Linear Regression

## Project Description

This project builds a complete house price prediction pipeline using the **California Housing dataset**. It uses Python, Pandas, NumPy, and Scikit-learn to load and inspect the data, split it into training and testing sets, apply feature scaling, train a Linear Regression model, evaluate its performance using MAE, MSE, RMSE, and R², compare the scaled model with a baseline model, and save the trained model using Joblib for future predictions.

The project is progressively improved as part of the **SkillPilot Machine Learning learning roadmap**.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Jupyter Notebook

## Machine Learning Pipeline

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
```

## Day 8: Feature Scaling and Model Comparison

As part of Day 8, the existing House Price Prediction project was extended with **feature scaling and baseline comparison**.

### Feature Scaling

The numerical features are scaled using Scikit-learn's `StandardScaler`.

Standard scaling transforms the features so that they have a comparable scale, which can help machine learning algorithms perform more effectively.

The scaling step is integrated into the machine learning pipeline:

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])
```

### Baseline Model

A **Linear Regression** model without feature scaling is first trained as the baseline.

Its R² score is recorded for comparison.

### Scaled Model

A second Linear Regression model is trained using a pipeline containing:

* `StandardScaler`
* `LinearRegression`

The R² score of the scaled model is then compared with the baseline R² score.

### R² Comparison

The project calculates:

```text
Baseline R²
Scaled Model R²
R² Difference
```

This helps determine whether adding feature scaling changed the model's performance.

## Categorical Encoding

The current **California Housing dataset contains numerical input features**, so One-Hot Encoding is not required for this particular dataset.

The project therefore focuses on **StandardScaler and pipeline-based preprocessing** for the California Housing data.

For a housing dataset containing categorical features such as `location` or `furnishingstatus`, `OneHotEncoder` can be added to the preprocessing pipeline using `ColumnTransformer`.

## Model

The project uses **Linear Regression** as the regression model.

The dataset is divided into:

* **80% training data**
* **20% testing data**

The model learns from the training data and is evaluated on the unseen testing data.

## Evaluation Metrics

The model is evaluated using:

* **MAE** — Mean Absolute Error
* **MSE** — Mean Squared Error
* **RMSE** — Root Mean Squared Error
* **R²** — R-squared score

The R² score is also used to compare the baseline model with the scaled pipeline.

## Model Persistence

After training, the complete scaled pipeline is saved using Joblib:

```text
house_price_model_scaled.pkl
```

The saved model can be loaded later using Joblib without training the model again.

## Project Goal

The goal of this project is to demonstrate a complete machine learning workflow, including:

* Data loading
* Data inspection
* Train/test splitting
* Baseline model creation
* Feature scaling
* Pipeline creation
* Linear Regression
* Model prediction
* Model evaluation
* R² comparison
* Model persistence

The project is being progressively improved through the **SkillPilot Machine Learning roadmap**.
