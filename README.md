\# House Price Prediction Using Linear Regression



\## Project Description



This project builds a baseline house price prediction pipeline using the California Housing dataset. It uses Python, Pandas, NumPy, and Scikit-learn to load and inspect the data, split it into training and testing sets, scale the features, train a Linear Regression model, evaluate its performance using MAE, MSE, RMSE, and R², and save the trained model using Joblib for future predictions.



\## Tools and Libraries



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Joblib

\* Jupyter Notebook



\## Machine Learning Pipeline



```text

California Housing Dataset

&#x20;         ↓

&#x20;    Data Loading

&#x20;         ↓

&#x20;   Data Inspection

&#x20;         ↓

&#x20;  Train/Test Split

&#x20;         ↓

&#x20;   Standard Scaling

&#x20;         ↓

&#x20;  Linear Regression

&#x20;         ↓

&#x20;     Prediction

&#x20;         ↓

&#x20;     Evaluation

&#x20;         ↓

&#x20;   Model Persistence

```



\## Model



The project uses \*\*Linear Regression\*\* as the baseline regression model.



The model is trained using 80% of the dataset and evaluated using the remaining 20%.



\## Evaluation Metrics



The model is evaluated using:



\* \*\*MAE\*\* — Mean Absolute Error

\* \*\*MSE\*\* — Mean Squared Error

\* \*\*RMSE\*\* — Root Mean Squared Error

\* \*\*R²\*\* — R-squared score



\## Model Persistence



After training, the complete pipeline is saved as:



```text

house\_price\_model.pkl

```



The saved model can be loaded later using Joblib without training the model again.



\## Project Goal



The goal is to demonstrate a complete baseline machine learning workflow, from data loading and preprocessing to regression, evaluation, and model persistence.



