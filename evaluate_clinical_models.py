import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from xgboost import XGBClassifier


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = r"dataset\clinical\processed_ckd_dataset.csv"

print("Loading clinical dataset...", flush=True)

df = pd.read_csv(DATA_PATH)

features = [
    "serum_creatinine",
    "gfr",
    "bun",
    "serum_calcium",
    "oxalate_levels",
    "urine_ph",
    "blood_pressure",
    "ana",
    "c3_c4",
    "hematuria",
    "smoking",
    "alcohol",
    "painkiller_usage",
    "family_history",
    "physical_activity",
    "diet",
    "water_intake",
    "weight_changes",
    "stress_level",
    "months"
]

X = df[features]
y = df["ckd_pred"].astype(int)

print("Total dataset records:", len(df), flush=True)


# ============================================================
# 2. CREATE FULL DATASET SPLIT FOR XGBOOST AND RANDOM FOREST
# ============================================================

X_train_full, X_test_full, y_train_full, y_test_full = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nFULL DATASET EVALUATION", flush=True)
print("Training records:", len(X_train_full), flush=True)
print("Testing records :", len(X_test_full), flush=True)


# ============================================================
# 3. CREATE 20,000-RECORD SAMPLE FOR SVM
# ============================================================

X_svm, _, y_svm, _ = train_test_split(
    X,
    y,
    train_size=20000,
    random_state=42,
    stratify=y
)

X_train_svm, X_test_svm, y_train_svm, y_test_svm = (
    train_test_split(
        X_svm,
        y_svm,
        test_size=0.20,
        random_state=42,
        stratify=y_svm
    )
)

print("\nSVM SAMPLE EVALUATION", flush=True)
print("Training records:", len(X_train_svm), flush=True)
print("Testing records :", len(X_test_svm), flush=True)


# ============================================================
# 4. SCALE DATA FOR SVM
# ============================================================

svm_scaler = StandardScaler()

X_train_svm_scaled = svm_scaler.fit_transform(X_train_svm)
X_test_svm_scaled = svm_scaler.transform(X_test_svm)


# ============================================================
# 5. EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(
    name,
    model,
    X_train_data,
    X_test_data,
    y_train_data,
    y_test_data,
    dataset_description
):

    print(f"\nTraining {name}...", flush=True)

    model.fit(X_train_data, y_train_data)

    predictions = model.predict(X_test_data)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test_data)[:, 1]
    else:
        probabilities = model.decision_function(X_test_data)

    metrics = {
        "Model": name,
        "Dataset": dataset_description,
        "Accuracy": accuracy_score(
            y_test_data, predictions
        ) * 100,
        "Precision": precision_score(
            y_test_data, predictions, zero_division=0
        ) * 100,
        "Recall": recall_score(
            y_test_data, predictions, zero_division=0
        ) * 100,
        "F1-score": f1_score(
            y_test_data, predictions, zero_division=0
        ) * 100,
        "ROC-AUC": roc_auc_score(
            y_test_data, probabilities
        ) * 100
    }

    results.append(metrics)

    print(f"{name} completed.", flush=True)


# ============================================================
# 6. SVM — 20,000 RECORDS
# ============================================================

svm_model = SVC(
    kernel="linear",
    probability=False,
    random_state=42
)

evaluate_model(
    "SVM",
    svm_model,
    X_train_svm_scaled,
    X_test_svm_scaled,
    y_train_svm,
    y_test_svm,
    "20,000-record sample"
)


# ============================================================
# 7. XGBOOST — FULL 500,000-RECORD DATASET
# ============================================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss",
    n_jobs=2,
    tree_method="hist"
)

evaluate_model(
    "XGBoost",
    xgb_model,
    X_train_full,
    X_test_full,
    y_train_full,
    y_test_full,
    "500,000-record dataset"
)


# ============================================================
# 8. RANDOM FOREST — FULL 500,000-RECORD DATASET
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=2
)

evaluate_model(
    "Random Forest - Proposed Methodology",
    rf_model,
    X_train_full,
    X_test_full,
    y_train_full,
    y_test_full,
    "500,000-record dataset"
)


# ============================================================
# 9. DISPLAY FINAL RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n")
print("=" * 110)
print("NEPHROPREDICTOR - CLINICAL MODEL PERFORMANCE")
print("=" * 110)

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.2f}%".format,
            "Precision": "{:.2f}%".format,
            "Recall": "{:.2f}%".format,
            "F1-score": "{:.2f}%".format,
            "ROC-AUC": "{:.2f}%".format
        }
    )
)

results_df.to_csv(
    "clinical_model_metrics.csv",
    index=False
)

print("\nResults saved to clinical_model_metrics.csv")
print("Evaluation completed successfully.", flush=True)