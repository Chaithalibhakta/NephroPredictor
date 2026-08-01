import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# Features and target
X = df.drop(["ckd_pred", "ckd_stage", "cluster"], axis=1)
y = df["ckd_pred"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load trained model
model = joblib.load("../models/best_ckd_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}\n")

# Classification Report
report = classification_report(y_test, y_pred)

print("Classification Report")
print("----------------------")
print(report)

# Save report to a text file
with open("../reports/classification_report.txt", "w") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n\n")
    file.write(report)

print("\nClassification report saved successfully!")