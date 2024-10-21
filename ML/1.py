# 1-6 have the same code just change the independent variable i.e (x) field

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
# Load the dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\vighnesh\ML\advertising\Advertising Budget and Sales.csv')


# 2. Define feature (Radio) and target (Sales)
X = df['Radio Ad Budget ($)']
y = df['Sales ($)']

# 3. Split the data into training and testing sets (manually)
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to numpy arrays for easier mathematical operations
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# 4. Calculate the coefficients using Ordinary Least Squares (OLS)
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)

numerator = np.sum((X_train - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train - X_mean)**2)
m = numerator / denominator  # Slope
b = y_mean - m * X_mean      # Intercept

print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")

# 5. Predict Sales on the test set
y_pred = m * X_test + b

# Display a few actual vs predicted values
comparison = pd.DataFrame({'Actual Sales': y_test, 'Predicted Sales': y_pred})
print("\nActual vs Predicted Sales:")
print(comparison.head())

# 6. Calculate RMSE
rmse = np.sqrt(np.mean((y_test - y_pred)**2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse}")
