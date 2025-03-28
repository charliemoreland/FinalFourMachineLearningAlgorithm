# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE  # For SMOTE resampling

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

# Step 5: Apply SMOTE to balance the dataset
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Step 6: Set up XGBoost model
xgb_model = xgb.XGBClassifier(random_state=42)

# Step 7: Set up the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'scale_pos_weight': [len(y_train) / (2 * sum(y_train == 1))]  # Adjust class weights
}

# Step 8: Perform GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train_res, y_train_res)

# Step 9: Train the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 10: Make predictions on the test set
y_pred = best_model.predict(X_test)

# Step 11: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)  # Accuracy score
conf_matrix = confusion_matrix(y_test, y_pred)  # Confusion matrix
class_report = classification_report(y_test, y_pred)  # Precision, Recall, F1-score
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])  # AUC score

# Print evaluation metrics
print(f"Best Parameters from GridSearchCV: {grid_search.best_params_}")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
print(f"ROC AUC: {roc_auc}")
