# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # You can choose any classifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load and Understand the Dataset
# Load the Iris dataset from sklearn
iris = datasets.load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: Iris species (Setosa, Versicolour, Virginica)

# Convert to DataFrame for better understanding
df = pd.DataFrame(data=np.c_[X, y], columns=iris.feature_names + ['species'])
print("First few rows of the Iris dataset:")
print(df.head())

# 2. Build a Classifier Without Applying PCA (e.g., SVM)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the classifier (SVM with linear kernel)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model (without PCA)
accuracy_no_pca = accuracy_score(y_test, y_pred)
conf_matrix_no_pca = confusion_matrix(y_test, y_pred)

print("\nAccuracy without PCA:", accuracy_no_pca)
print("\nConfusion Matrix without PCA:")
print(conf_matrix_no_pca)

# 3. Apply PCA for Dimensionality Reduction
# Reduce to 2 principal components for visualization and analysis
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Split the dataset into training and testing sets (after PCA transformation)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 4. Build the Same Classifier After Applying PCA
svm_model_pca = SVC(kernel='linear', random_state=42)
svm_model_pca.fit(X_train_pca, y_train_pca)

# Make predictions on the PCA-transformed test set
y_pred_pca = svm_model_pca.predict(X_test_pca)

# Evaluate the model (with PCA)
accuracy_with_pca = accuracy_score(y_test_pca, y_pred_pca)
conf_matrix_with_pca = confusion_matrix(y_test_pca, y_pred_pca)

print("\nAccuracy with PCA:", accuracy_with_pca)
print("\nConfusion Matrix with PCA:")
print(conf_matrix_with_pca)

# 5. Compare the Results
print("\nComparison of results:")
print(f"Accuracy without PCA: {accuracy_no_pca:.2f}")
print(f"Accuracy with PCA: {accuracy_with_pca:.2f}")
