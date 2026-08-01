import pandas as pd
import joblib
import matplotlib.pyplot as plt
from xgboost import plot_importance

# Load model
model = joblib.load("../models/best_ckd_model.pkl")

# Plot feature importance
plt.figure(figsize=(12,8))

plot_importance(
    model,
    importance_type="gain",
    max_num_features=15,
    height=0.6
)

plt.title("XGBoost Feature Importance")
plt.tight_layout()

plt.savefig("../reports/xgboost_feature_importance.png", dpi=300)

plt.show()

print("Feature Importance Graph Generated Successfully!")