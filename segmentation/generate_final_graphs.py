import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"
FEATURE_DIR = os.path.join(BASE_DIR, "segmentation", "features")
GRAPH_DIR = os.path.join(BASE_DIR, "segmentation", "features", "final_graphs")

os.makedirs(GRAPH_DIR, exist_ok=True)

# ============================================================
# 1. CLINICAL MODEL PERFORMANCE
# ============================================================

clinical_data = {
    "Accuracy": 99.998,
    "Precision": 100.0,
    "Recall": 99.996,
    "F1 Score": 99.998
}

plt.figure(figsize=(8, 5))
plt.bar(clinical_data.keys(), clinical_data.values())
plt.ylabel("Score (%)")
plt.title("Clinical Random Forest Performance")
plt.ylim(0, 105)

for i, value in enumerate(clinical_data.values()):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "clinical_model_performance.png"),
    dpi=300
)
plt.close()


# ============================================================
# 2. IMAGE MODEL COMPARISON
# ============================================================

image_data = {
    "Random Forest": [75.17, 77.66, 95.22, 85.55],
    "XGBoost": [74.16, 78.87, 90.87, 84.44]
}

metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

df = pd.DataFrame(
    image_data,
    index=metrics
)

ax = df.plot(
    kind="bar",
    figsize=(9, 6)
)

ax.set_ylabel("Score (%)")
ax.set_title("Ultrasound Classification Model Comparison")
ax.set_ylim(0, 105)
ax.legend(title="Model")

plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "image_model_comparison.png"),
    dpi=300
)
plt.close()


# ============================================================
# 3. U-NET SEGMENTATION PERFORMANCE
# ============================================================

unet_data = {
    "Dice": 69.8638,
    "IoU": 58.7270,
    "Precision": 86.3296,
    "Recall": 64.1923
}

plt.figure(figsize=(8, 5))
plt.bar(unet_data.keys(), unet_data.values())

plt.ylabel("Score (%)")
plt.title("U-Net Kidney Segmentation Performance")
plt.ylim(0, 105)

for i, value in enumerate(unet_data.values()):
    plt.text(i, value + 1, f"{value:.2f}%", ha="center")

plt.tight_layout()

plt.savefig(
    os.path.join(GRAPH_DIR, "unet_segmentation_performance.png"),
    dpi=300
)
plt.close()


print("=" * 60)
print("FINAL GRAPHS GENERATED SUCCESSFULLY")
print("=" * 60)
print()
print("Saved in:")
print(GRAPH_DIR)
print()
print("Files:")
print("1. clinical_model_performance.png")
print("2. image_model_comparison.png")
print("3. unet_segmentation_performance.png")