import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
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

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("Shape:", df.shape)

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
smote = SMOTE(random_state=42)

X_resampled, y_resampled = smote.fit_resample(X, y)

print("\nBalanced Dataset Distribution")
print(pd.Series(y_resampled).value_counts())

# ==========================================
# Train Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.20,
    random_state=42,
    stratify=y_resampled
)

# ==========================================
# Random Forest
# ==========================================
print("\n==============================")
print("Training Random Forest")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_pred)

rf_auc = roc_auc_score(y_test, rf_prob)

print("\nRandom Forest Accuracy:", round(rf_acc * 100, 2), "%")

print("Random Forest AUC:", round(rf_auc, 4))

print("\nClassification Report")

print(classification_report(y_test, rf_pred))

print("Confusion Matrix")

print(confusion_matrix(y_test, rf_pred))

# ==========================================
# XGBoost
# ==========================================
print("\n==============================")
print("Training XGBoost")
print("==============================")

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_prob = xgb.predict_proba(X_test)[:, 1]

xgb_acc = accuracy_score(y_test, xgb_pred)

xgb_auc = roc_auc_score(y_test, xgb_prob)

print("\nXGBoost Accuracy:", round(xgb_acc * 100, 2), "%")

print("XGBoost AUC:", round(xgb_auc, 4))

print("\nClassification Report")

print(classification_report(y_test, xgb_pred))

print("Confusion Matrix")

print(confusion_matrix(y_test, xgb_pred))
# ==========================================
# Model Comparison
# ==========================================
print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")
print(f"Random Forest AUC      : {rf_auc:.4f}")

print(f"XGBoost Accuracy       : {xgb_acc * 100:.2f}%")
print(f"XGBoost AUC            : {xgb_auc:.4f}")

# ==========================================
# Select Best Model
# ==========================================
if xgb_acc >= rf_acc:
    best_model = xgb
    best_name = "XGBoost"
    best_accuracy = xgb_acc
else:
    best_model = rf
    best_name = "Random Forest"
    best_accuracy = rf_acc

# ==========================================
# Save Best Model
# ==========================================
joblib.dump(best_model, "../models/best_ckd_model.pkl")

print("\n===================================")
print("BEST MODEL")
print("===================================")

print("Best Model :", best_name)
print("Accuracy   :", round(best_accuracy * 100, 2), "%")

print("\nModel saved successfully!")

# ==========================================
# Verify Saved Model
# ==========================================
loaded_model = joblib.load("../models/best_ckd_model.pkl")

print("\nLoaded Model Type:")
print(type(loaded_model))

print("\nClasses:")
print(loaded_model.classes_)

print("\nPrediction Check")

sample = X_test.iloc[[0]]

print("Actual Label    :", y_test.iloc[0])
print("Predicted Label :", loaded_model.predict(sample)[0])
print("Probabilities   :", loaded_model.predict_proba(sample)[0])

print("\n===================================")
print("TRAINING COMPLETED SUCCESSFULLY")
print("===================================")