import os
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
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
print("LOADING DATA")
print("=" * 70)

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))


# ============================================================
# 3. FEATURES
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


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

healthy_count = (y_train == 0).sum()
pathological_count = (y_train == 1).sum()

print("\nTraining class distribution:")
print("Healthy:", healthy_count)
print("Pathological:", pathological_count)


# ============================================================
# 5. XGBOOST CLASS WEIGHT
# ============================================================

# We want to give MORE importance to the minority class:
# Healthy = 0
#
# XGBoost's scale_pos_weight applies to class 1,
# so instead of using it directly, we use sample_weight.

healthy_weight = (
    pathological_count / healthy_count
)

pathological_weight = 1.0

sample_weights = np.where(
    y_train == 0,
    healthy_weight,
    pathological_weight
)

print("\nClass weights:")
print(
    "Healthy weight:",
    round(healthy_weight, 4)
)

print(
    "Pathological weight:",
    pathological_weight
)


# ============================================================
# 6. RANDOM FOREST - BALANCED
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING BALANCED RANDOM FOREST")
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

rf_pred = rf_model.predict(
    X_val
)

print("Random Forest completed.")


# ============================================================
# 7. RANDOM FOREST METRICS
# ============================================================

rf_accuracy = accuracy_score(
    y_val,
    rf_pred
)

rf_balanced_accuracy = balanced_accuracy_score(
    y_val,
    rf_pred
)

rf_precision_macro = precision_score(
    y_val,
    rf_pred,
    average="macro",
    zero_division=0
)

rf_recall_macro = recall_score(
    y_val,
    rf_pred,
    average="macro",
    zero_division=0
)

rf_f1_macro = f1_score(
    y_val,
    rf_pred,
    average="macro",
    zero_division=0
)

rf_f1_weighted = f1_score(
    y_val,
    rf_pred,
    average="weighted",
    zero_division=0
)

rf_cm = confusion_matrix(
    y_val,
    rf_pred
)


print("\n" + "=" * 70)
print("BALANCED RANDOM FOREST RESULTS")
print("=" * 70)

print(
    f"Accuracy          : {rf_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {rf_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : {rf_precision_macro:.4f}"
)

print(
    f"Macro Recall      : {rf_recall_macro:.4f}"
)

print(
    f"Macro F1          : {rf_f1_macro:.4f}"
)

print(
    f"Weighted F1       : {rf_f1_weighted:.4f}"
)

print("\nConfusion Matrix:")
print(rf_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        rf_pred,
        target_names=[
            "Healthy",
            "Pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 8. XGBOOST - BALANCED
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING BALANCED XGBOOST")
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
    y_train,
    sample_weight=sample_weights
)

xgb_pred = xgb_model.predict(
    X_val
)

print("XGBoost completed.")


# ============================================================
# 9. XGBOOST METRICS
# ============================================================

xgb_accuracy = accuracy_score(
    y_val,
    xgb_pred
)

xgb_balanced_accuracy = balanced_accuracy_score(
    y_val,
    xgb_pred
)

xgb_precision_macro = precision_score(
    y_val,
    xgb_pred,
    average="macro",
    zero_division=0
)

xgb_recall_macro = recall_score(
    y_val,
    xgb_pred,
    average="macro",
    zero_division=0
)

xgb_f1_macro = f1_score(
    y_val,
    xgb_pred,
    average="macro",
    zero_division=0
)

xgb_f1_weighted = f1_score(
    y_val,
    xgb_pred,
    average="weighted",
    zero_division=0
)

xgb_cm = confusion_matrix(
    y_val,
    xgb_pred
)


print("\n" + "=" * 70)
print("BALANCED XGBOOST RESULTS")
print("=" * 70)

print(
    f"Accuracy          : {xgb_accuracy:.4f}"
)

print(
    f"Balanced Accuracy : {xgb_balanced_accuracy:.4f}"
)

print(
    f"Macro Precision   : {xgb_precision_macro:.4f}"
)

print(
    f"Macro Recall      : {xgb_recall_macro:.4f}"
)

print(
    f"Macro F1          : {xgb_f1_macro:.4f}"
)

print(
    f"Weighted F1       : {xgb_f1_weighted:.4f}"
)

print("\nConfusion Matrix:")
print(xgb_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        xgb_pred,
        target_names=[
            "Healthy",
            "Pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. COMPARISON
# ============================================================

comparison = pd.DataFrame({

    "Model": [
        "Balanced Random Forest",
        "Balanced XGBoost"
    ],

    "Accuracy": [
        rf_accuracy,
        xgb_accuracy
    ],

    "Balanced_Accuracy": [
        rf_balanced_accuracy,
        xgb_balanced_accuracy
    ],

    "Macro_Precision": [
        rf_precision_macro,
        xgb_precision_macro
    ],

    "Macro_Recall": [
        rf_recall_macro,
        xgb_recall_macro
    ],

    "Macro_F1": [
        rf_f1_macro,
        xgb_f1_macro
    ],

    "Weighted_F1": [
        rf_f1_weighted,
        xgb_f1_weighted
    ]
})


# ============================================================
# 11. DISPLAY COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("BALANCED MODEL COMPARISON")
print("=" * 70)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

output_file = os.path.join(
    FEATURE_DIR,
    "balanced_model_comparison.csv"
)

comparison.to_csv(
    output_file,
    index=False
)


# ============================================================
# 13. SAVE CONFUSION MATRICES
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

rf_cm_df.to_csv(
    os.path.join(
        FEATURE_DIR,
        "balanced_random_forest_confusion_matrix.csv"
    )
)

xgb_cm_df.to_csv(
    os.path.join(
        FEATURE_DIR,
        "balanced_xgboost_confusion_matrix.csv"
    )
)


# ============================================================
# 14. FINAL
# ============================================================

print("\n")
print("=" * 70)
print("BALANCED COMPARISON COMPLETED")
print("=" * 70)

print(
    "Results saved to:"
)

print(
    output_file
)

print("=" * 70)