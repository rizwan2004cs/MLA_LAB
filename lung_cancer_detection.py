# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data generation (replace this with your actual dataset)
data = {
    'age': [45, 50, 34, 67, 52, 30, 56, 63, 45, 29],
    'smoking_status': [1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'cough_duration': [10, 20, 5, 15, 8, 3, 12, 25, 10, 6],
    'breathlessness': [1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'chronic_disease': [1, 1, 0, 1, 0, 0, 1, 1, 0, 0],
    'lung_disease': [1, 1, 0, 1, 0, 0, 1, 1, 0, 0]
}

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df.drop(columns=['lung_disease'])
y = df['lung_disease']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling (important for models like SVM, k-NN, etc. but optional here for RandomForest)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)

# Get the probabilities for the positive class (for ROC curve)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

# Compute the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute the AUC score
roc_auc = roc_auc_score(y_test, y_prob)

# Plot the ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Additional evaluation metrics (for completeness)
accuracy = accuracy_score(y_test, clf.predict(X_test_scaled))
conf_matrix = confusion_matrix(y_test, clf.predict(X_test_scaled))
class_report = classification_report(y_test, clf.predict(X_test_scaled))

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)
