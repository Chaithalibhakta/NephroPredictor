import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 50)
print("LOADING DATASET")
print("=" * 50)

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# -----------------------------
# Features
# -----------------------------
X = df.drop(["ckd_pred", "ckd_stage"], axis=1)

# -----------------------------
# Target
# -----------------------------
y = df["ckd_stage"].astype(int)

print("\nDataset Shape:", df.shape)

print("\nStage Distribution")
print(y.value_counts().sort_index())

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------
# Train Random Forest
# -----------------------------
print("\nTraining Stage Model...")

stage_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

stage_model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
y_pred = stage_model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
joblib.dump(stage_model, "../models/stage_model.pkl")

print("\nStage Model Saved Successfully!")