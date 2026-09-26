# 1. Import libraries
import pandas as pd
import numpy as np
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 2. Load California Housing dataset
data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

print("Dataset loaded")
print(X.head())


# 3. Basic inspection
print("\nDataset shape:", X.shape)
print("\nMissing values:")
print(X.isnull().sum())


# 4. Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# 5. Create Pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("regression", LinearRegression())
])


# 6. Train the model
model.fit(X_train, y_train)

print("\nModel trained successfully!")


# 7. Make predictions
y_pred = model.predict(X_test)


# 8. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2  :", r2)


# 9. Save the model
joblib.dump(model, "house_price_model.pkl")

print("\nModel saved successfully!")