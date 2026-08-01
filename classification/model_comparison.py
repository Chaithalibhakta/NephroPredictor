import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# Features
X = df.drop(["ckd_pred", "ckd_stage"], axis=1)

# Target
y = df["ckd_pred"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ---------------- Random Forest ----------------
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# ---------------- XGBoost ----------------
xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)

xgb_acc = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_acc)

# ---------------- Save Best Model ----------------
if rf_acc >= xgb_acc:
    joblib.dump(rf, "../models/best_ckd_model.pkl")
    print("Best Model: Random Forest")
else:
    joblib.dump(xgb, "../models/best_ckd_model.pkl")
    print("Best Model: XGBoost")

print("Best model saved successfully!")