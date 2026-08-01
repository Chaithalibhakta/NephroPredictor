import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

# Features and target
X = df.drop(["ckd_pred", "ckd_stage", "cluster"], axis=1)
y = df["ckd_pred"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Load best trained model
model = joblib.load("../models/best_ckd_model.pkl")

# Predict
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.savefig("../reports/confusion_matrix.png", dpi=300)
plt.show()

print("Confusion Matrix generated successfully!")