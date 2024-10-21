import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\vighnesh\ML\car\car data.csv')

# 2. Inspect the data
print(df.info())  # Check the data types

# 3. Convert 'year_bought' to numeric, coerce errors (convert non-numeric values to NaN)
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Selling_Price'] = pd.to_numeric(df['Selling_Price'], errors='coerce')


# Check if any NaN values were created due to non-numeric data
print(df['Year'].isnull().sum())

# Optionally, drop rows with NaN values in 'year_bought' and 'selling_price'
df.dropna(subset=['Year', 'Selling_Price'], inplace=True)

# Convert feature and target to numpy arrays
X = df['Year'].to_numpy()
y = df['Selling_Price'].to_numpy()

# 4. Split the data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 5. Calculate the coefficients using Ordinary Least Squares (OLS)
X_mean = np.mean(X_train)
y_mean = np.mean(y_train)

# Calculate the slope (m) and intercept (b)
numerator = np.sum((X_train - X_mean) * (y_train - y_mean))
denominator = np.sum((X_train - X_mean)**2)
m = numerator / denominator  # Slope
b = y_mean - m * X_mean      # Intercept

print(f"\nSlope (m): {m}")
print(f"Intercept (b): {b}")

# 6. Predict Selling Price on the test set
y_pred = m * X_test + b

# 7. Compare actual vs predicted values
comparison = pd.DataFrame({'Actual Selling Price': y_test, 'Predicted Selling Price': y_pred})
print("\nActual vs Predicted Selling Prices:")
print(comparison.head())

# 8. Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
print(f"\nRoot Mean Squared Error (RMSE): {rmse:.2f}")
