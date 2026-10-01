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

TEST_FILE = os.path.join(
    FEATURE_DIR,
    "test_image_features.csv"
)

OUTPUT_DIR = FEATURE_DIR

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. FEATURE COLUMNS
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

TARGET_COLUMN = "diagnosis"


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING IMAGE FEATURE DATA")
print("=" * 70)

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Test samples:",
    len(test_df)
)

print()


# ============================================================
# 4. PREPARE X AND Y
# ============================================================

X_train = train_df[
    FEATURE_COLUMNS
]

y_train = train_df[
    TARGET_COLUMN
]

X_test = test_df[
    FEATURE_COLUMNS
]

y_test = test_df[
    TARGET_COLUMN
]


print("Training class distribution:")
print(y_train.value_counts())

print()

print("Test class distribution:")
print(y_test.value_counts())

print()


# ============================================================
# 5. RANDOM FOREST
# ============================================================

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
    X_test
)

rf_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 6. RANDOM FOREST METRICS
# ============================================================

rf_accuracy = accuracy_score(
    y_test,
    rf_predictions
)

rf_precision = precision_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_predictions,
    zero_division=0
)

rf_balanced_accuracy = balanced_accuracy_score(
    y_test,
    rf_predictions
)

rf_cm = confusion_matrix(
    y_test,
    rf_predictions
)


print("\nRandom Forest Results")
print("-" * 50)

print(
    f"Accuracy:           {rf_accuracy:.4f}"
)

print(
    f"Precision:          {rf_precision:.4f}"
)

print(
    f"Recall:             {rf_recall:.4f}"
)

print(
    f"F1 Score:           {rf_f1:.4f}"
)

print(
    f"Balanced Accuracy:  {rf_balanced_accuracy:.4f}"
)

print("\nConfusion Matrix:")

print(rf_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_predictions,
        target_names=[
            "healthy",
            "pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 7. SAVE RANDOM FOREST CONFUSION MATRIX
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

rf_cm_path = os.path.join(
    OUTPUT_DIR,
    "test_random_forest_confusion_matrix.csv"
)

rf_cm_df.to_csv(
    rf_cm_path
)


# ============================================================
# 8. XGBOOST
# ============================================================

print("=" * 70)
print("TRAINING XGBOOST")
print("=" * 70)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train,
    y_train
)

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


# ============================================================
# 9. XGBOOST METRICS
# ============================================================

xgb_accuracy = accuracy_score(
    y_test,
    xgb_predictions
)

xgb_precision = precision_score(
    y_test,
    xgb_predictions,
    zero_division=0
)

xgb_recall = recall_score(
    y_test,
    xgb_predictions,
    zero_division=0
)

xgb_f1 = f1_score(
    y_test,
    xgb_predictions,
    zero_division=0
)

xgb_balanced_accuracy = balanced_accuracy_score(
    y_test,
    xgb_predictions
)

xgb_cm = confusion_matrix(
    y_test,
    xgb_predictions
)


print("\nXGBoost Results")
print("-" * 50)

print(
    f"Accuracy:           {xgb_accuracy:.4f}"
)

print(
    f"Precision:          {xgb_precision:.4f}"
)

print(
    f"Recall:             {xgb_recall:.4f}"
)

print(
    f"F1 Score:           {xgb_f1:.4f}"
)

print(
    f"Balanced Accuracy:  {xgb_balanced_accuracy:.4f}"
)

print("\nConfusion Matrix:")

print(xgb_cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        xgb_predictions,
        target_names=[
            "healthy",
            "pathological"
        ],
        zero_division=0
    )
)


# ============================================================
# 10. SAVE XGBOOST CONFUSION MATRIX
# ============================================================

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

xgb_cm_path = os.path.join(
    OUTPUT_DIR,
    "test_xgboost_confusion_matrix.csv"
)

xgb_cm_df.to_csv(
    xgb_cm_path
)


# ============================================================
# 11. MODEL COMPARISON
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

    "F1": [
        rf_f1,
        xgb_f1
    ],

    "Balanced Accuracy": [
        rf_balanced_accuracy,
        xgb_balanced_accuracy
    ]
})


comparison_path = os.path.join(
    OUTPUT_DIR,
    "test_image_model_comparison.csv"
)

comparison.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = test_df[
    [
        "image_id",
        "diagnosis",
        "class_name",
        "image_path"
    ]
].copy()

predictions_df[
    "rf_prediction"
] = rf_predictions

predictions_df[
    "rf_probability"
] = rf_probabilities

predictions_df[
    "xgb_prediction"
] = xgb_predictions

predictions_df[
    "xgb_probability"
] = xgb_probabilities


predictions_path = os.path.join(
    OUTPUT_DIR,
    "test_image_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("IMAGE TEST EVALUATION COMPLETED")
print("=" * 70)

print("\nModel Comparison:")
print(comparison.to_string(index=False))

print("\nSaved files:")

print(
    "1.",
    comparison_path
)

print(
    "2.",
    rf_cm_path
)

print(
    "3.",
    xgb_cm_path
)

print(
    "4.",
    predictions_path
)

print("=" * 70)