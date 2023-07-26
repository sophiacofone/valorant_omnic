import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_selection import SelectFromModel

# Perform feature selection using L1-regularized logistic regression
logreg = LogisticRegressionCV(penalty='l1', solver='liblinear', cv=5)
logreg.fit(X_train_scaled, y_train)

# Select features based on non-zero coefficients
selector = SelectFromModel(logreg, threshold='median')
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Get the selected feature names
selected_feature_names = X.columns[selector.get_support()]

selected_feature_names

# Train and evaluate the model on the selected features
logreg_selected = LogisticRegressionCV(solver='lbfgs', cv=5)
logreg_selected.fit(X_train_selected, y_train)

# Calculate metrics on the training set
train_predictions = logreg_selected.predict(X_train_selected)
train_accuracy = accuracy_score(y_train, train_predictions)
train_precision = precision_score(y_train, train_predictions)
train_recall = recall_score(y_train, train_predictions)
train_f1 = f1_score(y_train, train_predictions)

print("Training Set Metrics:")
print(f'Training Accuracy: {train_accuracy}')
print(f'Training Precision: {train_precision}')
print(f'Training Recall: {train_recall}')
print(f'Training F1 score: {train_f1}')

# Calculate metrics on the test set
predictions = logreg_selected.predict(X_test_selected)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("Test Set Metrics:")
print(f'Test Accuracy: {accuracy}')
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')
print(f'Test F1 score: {f1}')