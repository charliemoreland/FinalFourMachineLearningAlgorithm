# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load the Kaggle dataset
# Replace 'your_dataset.csv' with the path to your Kaggle dataset
data = pd.read_csv('cbb.csv')

# Step 2: Drop unwanted columns (TEAM, CONF, G, POSTSEASON)
data = data.drop(['TEAM', 'CONF', 'G', 'POSTSEASON', 'SEED'], axis=1)

# Step 3: Select features (X) and target variable (y)
# Assuming 'F4' is the column you want to predict, and the rest are features
X = data.drop('F4', axis=1)  # All columns except 'F4' will be used as features
y = data['F4']  # 'F4' is the target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the Random Forest model
#model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Step 6: Train the model on the training data
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
class_report = classification_report(y_test, y_pred)  # Precision, Recall, F1-score
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])  # AUC score

# Print evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
print(f"ROC AUC: {roc_auc}")
