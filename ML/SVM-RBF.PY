import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\vighnesh\ML\Social network dataset\Social_Network_Ads.csv')

# 2. Encode the 'Gender' column (0 for Female, 1 for Male)
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Encode gender

# 3. Prepare the features and target
X = df[['Gender', 'Age', 'EstimatedSalary']]  # Features
y = df['Purchased']  # Target

# 4. Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the SVM model with a linear kernel
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# 6. Make predictions on the test set
y_pred_test = model.predict(X_test)

# 7. Evaluate the model (accuracy, precision, recall, F1, and confusion matrix)
test_acc = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
conf_matrix = confusion_matrix(y_test, y_pred_test)

# Output the evaluation metrics
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)

# 8. Output "Yes" or "No" for car purchase
y_pred_output = ['Yes' if pred == 1 else 'No' for pred in y_pred_test]
print("\nPrediction for whether a user purchased a car:")
print(y_pred_output)
