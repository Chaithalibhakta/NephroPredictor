
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


# ==========================================
# Load Processed Dataset
# ==========================================

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

print("=" * 60)
print("NEPHROPREDICTOR - CLINICAL MODEL EVALUATION")
print("=" * 60)

print("Dataset shape:", df.shape)


# ==========================================
# Features & Target
# ==========================================

X = df.drop(["ckd_pred", "ckd_stage"], axis=1)
y = df["ckd_pred"]

print("\nOriginal Dataset Distribution")
print(y.value_counts())


# ==========================================
# Apply SMOTE
# ==========================================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nBalanced Dataset Distribution")
print(pd.Series(y_resampled).value_counts())


# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.20,
    random_state=42,
    stratify=y_resampled
)

print("\nTraining records:", len(X_train))
print("Testing records :", len(X_test))


# ==========================================
# Helper Function for Metrics
# ==========================================

def calculate_metrics(y_true, predictions, probabilities):

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }


def display_metrics(model_name, metrics):

    print(f"\n{model_name} PERFORMANCE")
    print("-" * 40)

    for metric_name, value in metrics.items():
        print(f"{metric_name:<12}: {value * 100:.2f}%")


# ==========================================
# Random Forest
# ==========================================

print("\nTraining Random Forest...")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

rf_metrics = calculate_metrics(
    y_test, rf_pred, rf_prob
)

display_metrics("RANDOM FOREST", rf_metrics)

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_pred))

print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_pred))


# ==========================================
# XGBoost
# ==========================================

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

xgb_metrics = calculate_metrics(
    y_test, xgb_pred, xgb_prob
)

display_metrics("XGBOOST", xgb_metrics)

print("\nXGBoost Classification Report")
print(classification_report(y_test, xgb_pred))

print("XGBoost Confusion Matrix")
print(confusion_matrix(y_test, xgb_pred))


# ==========================================
# Model Comparison
# ==========================================

print("\n")
print("=" * 85)
print("FINAL MODEL COMPARISON")
print("=" * 85)

results = pd.DataFrame([
    {"Model": "Random Forest", **rf_metrics},
    {"Model": "XGBoost", **xgb_metrics}
])

# Convert metric values to percentages for display.
display_results = results.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score",
    "ROC-AUC"
]

for column in metric_columns:
    display_results[column] = display_results[column].map(
        lambda value: f"{value * 100:.2f}%"
    )

print(display_results.to_string(index=False))


# ==========================================
# Identify Best Model
# ==========================================

if xgb_metrics["Accuracy"] >= rf_metrics["Accuracy"]:
    best_name = "XGBoost"
    best_accuracy = xgb_metrics["Accuracy"]
else:
    best_name = "Random Forest"
    best_accuracy = rf_metrics["Accuracy"]


print("\nBest Model in This Evaluation:", best_name)
print(f"Accuracy: {best_accuracy * 100:.2f}%")


# ==========================================
# Save Evaluation Metrics
# ==========================================

results.to_csv(
    "model_comparison_metrics.csv",
    index=False
)

print("\nMetrics saved to:")
print("classification/model_comparison_metrics.csv")


# ==========================================
# Protect Existing Model
# ==========================================

print("\nExisting deployed model has NOT been modified.")
print("File protected: ../models/best_ckd_model.pkl")

print("\nEvaluation completed successfully.")

