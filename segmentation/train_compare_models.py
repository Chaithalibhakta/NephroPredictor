import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "segmentation",
    "features"
)

TRAIN_FILE = os.path.join(
    FEATURE_DIR,
    "train_image_features.csv"
)

VAL_FILE = os.path.join(
    FEATURE_DIR,
    "val_image_features.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING IMAGE FEATURES")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# 3. SELECT FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "kidney_area",
    "kidney_perimeter",
    "kidney_width",
    "kidney_height",
    "aspect_ratio",
    "circularity",
    "mean_intensity",
    "std_intensity",
    "min_intensity",
    "max_intensity",
    "median_intensity",
    "roi_mean",
    "roi_std"
]


X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["diagnosis"]

X_val = val_df[FEATURE_COLUMNS]
y_val = val_df["diagnosis"]


print("\nNumber of features:", len(FEATURE_COLUMNS))

print("\nFeatures used:")
for feature in FEATURE_COLUMNS:
    print(" -", feature)


# ============================================================
# 4. CHECK DATA
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("Training shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())


# ============================================================
# 5. RANDOM FOREST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_val
)

print("Random Forest training completed.")


# ============================================================
# 6. RANDOM FOREST EVALUATION
# ============================================================

rf_accuracy = accuracy_score(
    y_val,
    rf_predictions
)

rf_precision = precision_score(
    y_val,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_val,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_val,
    rf_predictions,
    zero_division=0
)

rf_cm = confusion_matrix(
    y_val,
    rf_predictions
)


print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {rf_accuracy:.4f}"
)

print(
    f"Precision : {rf_precision:.4f}"
)

print(
    f"Recall    : {rf_recall:.4f}"
)

print(
    f"F1-Score  : {rf_f1:.4f}"
)

print("\nConfusion Matrix:")
print(rf_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        rf_predictions,
        target_names=[
            "Healthy",
            "Pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 7. XGBOOST
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_val
)

print("XGBoost training completed.")


# ============================================================
# 8. XGBOOST EVALUATION
# ============================================================

xgb_accuracy = accuracy_score(
    y_val,
    xgb_predictions
)

xgb_precision = precision_score(
    y_val,
    xgb_predictions,
    zero_division=0
)

xgb_recall = recall_score(
    y_val,
    xgb_predictions,
    zero_division=0
)

xgb_f1 = f1_score(
    y_val,
    xgb_predictions,
    zero_division=0
)

xgb_cm = confusion_matrix(
    y_val,
    xgb_predictions
)


print("\n" + "=" * 70)
print("XGBOOST RESULTS")
print("=" * 70)

print(
    f"Accuracy  : {xgb_accuracy:.4f}"
)

print(
    f"Precision : {xgb_precision:.4f}"
)

print(
    f"Recall    : {xgb_recall:.4f}"
)

print(
    f"F1-Score  : {xgb_f1:.4f}"
)

print("\nConfusion Matrix:")
print(xgb_cm)

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        xgb_predictions,
        target_names=[
            "Healthy",
            "Pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Random Forest",
        "XGBoost"
    ],

    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],

    "Precision": [
        rf_precision,
        xgb_precision
    ],

    "Recall": [
        rf_recall,
        xgb_recall
    ],

    "F1_Score": [
        rf_f1,
        xgb_f1
    ]
})


# ============================================================
# 10. DISPLAY COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 11. SAVE COMPARISON
# ============================================================

comparison_file = os.path.join(
    FEATURE_DIR,
    "image_model_comparison.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# 12. SAVE CONFUSION MATRICES
# ============================================================

rf_cm_df = pd.DataFrame(
    rf_cm,
    index=[
        "Actual Healthy",
        "Actual Pathological"
    ],
    columns=[
        "Predicted Healthy",
        "Predicted Pathological"
    ]
)

xgb_cm_df = pd.DataFrame(
    xgb_cm,
    index=[
        "Actual Healthy",
        "Actual Pathological"
    ],
    columns=[
        "Predicted Healthy",
        "Predicted Pathological"
    ]
)


rf_cm_file = os.path.join(
    FEATURE_DIR,
    "random_forest_confusion_matrix.csv"
)

xgb_cm_file = os.path.join(
    FEATURE_DIR,
    "xgboost_confusion_matrix.csv"
)

rf_cm_df.to_csv(
    rf_cm_file
)

xgb_cm_df.to_csv(
    xgb_cm_file
)


# ============================================================
# 13. FEATURE IMPORTANCE
# ============================================================

rf_importance = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": rf_model.feature_importances_

}).sort_values(
    by="importance",
    ascending=False
)


xgb_importance = pd.DataFrame({

    "feature": FEATURE_COLUMNS,

    "importance": xgb_model.feature_importances_

}).sort_values(
    by="importance",
    ascending=False
)


rf_importance_file = os.path.join(
    FEATURE_DIR,
    "random_forest_feature_importance.csv"
)

xgb_importance_file = os.path.join(
    FEATURE_DIR,
    "xgboost_feature_importance.csv"
)


rf_importance.to_csv(
    rf_importance_file,
    index=False
)

xgb_importance.to_csv(
    xgb_importance_file,
    index=False
)


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print("\nSaved files:")

print(
    "1.",
    comparison_file
)

print(
    "2.",
    rf_cm_file
)

print(
    "3.",
    xgb_cm_file
)

print(
    "4.",
    rf_importance_file
)

print(
    "5.",
    xgb_importance_file
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)