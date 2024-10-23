import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Bagging Classifier
model_bagging = BaggingClassifier(n_estimators=100, random_state=423)
model_bagging.fit(X_train, y_train)

# 2. AdaBoost Classifier
model_AdaBoost = AdaBoostClassifier(n_estimators=100, random_state=423)
model_AdaBoost.fit(X_train, y_train)

# 3. Random Forest Classifier
model_RandomForest = RandomForestClassifier(n_estimators=100, random_state=423)
model_RandomForest.fit(X_train, y_train)

# 4. Gradient Boosting Classifier
model_GradientBoosting = GradientBoostingClassifier(n_estimators=100, random_state=423)
model_GradientBoosting.fit(X_train, y_train)

# 5. Decision Tree Classifier (baseline)
model_decision_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, min_samples_split=2, random_state=42)
model_decision_tree.fit(X_train, y_train)

# Evaluate each model on the test set
models = {
    "Bagging Classifier": model_bagging,
    "AdaBoost Classifier": model_AdaBoost,
    "Random Forest Classifier": model_RandomForest,
    "Gradient Boosting Classifier": model_GradientBoosting,
    "Decision Tree Classifier": model_decision_tree
}

for name, model in models.items():
    # Make predictions on the test set
    y_pred_test = model.predict(X_test)
    
    # Evaluate the model
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, average='macro')
    recall = recall_score(y_test, y_pred_test, average='macro')
    f1 = f1_score(y_test, y_pred_test, average='macro')
    conf_matrix = confusion_matrix(y_test, y_pred_test)

    # Output the evaluation metrics
    print(f"----- {name} -----")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\n")
