import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

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

# Probability predictions
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Calculate AUC
auc_score = roc_auc_score(y_test, y_prob)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0,1], [0,1], linestyle="--", color="red")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

# Save graph
plt.savefig("../reports/roc_curve.png", dpi=300)

plt.show()

print(f"AUC Score: {auc_score:.4f}")
print("ROC Curve generated successfully!")