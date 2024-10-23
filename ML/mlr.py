#  mlr

import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\vighnesh\ML\advertising\Advertising Budget and Sales.csv')

# 2. Define features (Radio and TV) and target (Sales)
X = df[['Radio Ad Budget ($)', 'TV Ad Budget ($)']].values  # Multiple features
y = df['Sales ($)'].values

# 3. Add a column of ones to X to account for the intercept (bias term)
X = np.c_[np.ones(X.shape[0]), X]  # Adds a column of 1's to the feature matrix

# 4. Split the data into training and testing sets manually (80% training, 20% testing)
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Compute the coefficients (beta) using the normal equation: beta = (X^T * X)^-1 * X^T * y
beta = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# Separate the intercept and feature coefficients
intercept = beta[0]  # Intercept (the first value of beta)
coef_radio = beta[1]  # Coefficient for 'Radio Ad Budget'
coef_tv = beta[2]     # Coefficient for 'TV Ad Budget'

# Print the intercept and coefficients separately
print(f"Intercept: {intercept}")
print(f"Coefficient for Radio Ad Budget: {coef_radio}")
print(f"Coefficient for TV Ad Budget: {coef_tv}")

# 6. Predict sales for the test set
y_pred = X_test.dot(beta)

# 7. Display actual vs predicted values for the test set
comparison = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
print("\nActual vs Predicted Sales:")
print(comparison.head())

# 8. Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")
