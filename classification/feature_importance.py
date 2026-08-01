import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# Features
X = df.drop(["ckd_pred", "ckd_stage", "cluster"], axis=1)

# Load trained model
model = joblib.load("../models/best_ckd_model.pkl")

# Check if model supports feature importance
if hasattr(model, "feature_importances_"):

    importances = model.feature_importances_

    feature_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    })

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    print(feature_importance)

    plt.figure(figsize=(10,8))

    plt.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.xlabel("Importance Score")
    plt.ylabel("Clinical Features")
    plt.title("Feature Importance for CKD Prediction")

    plt.gca().invert_yaxis()

    plt.tight_layout()

    plt.savefig("../reports/feature_importance.png", dpi=300)

    plt.show()

    print("Feature Importance Graph generated successfully!")

else:
    print("This model does not support feature importance.")