import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

CLINICAL_FILE = os.path.join(
    BASE_DIR,
    "features",
    "clinical_feature_importance.csv"
)

IMAGE_FILE = os.path.join(
    BASE_DIR,
    "segmentation",
    "features",
    "image_feature_importance.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "segmentation",
    "features",
    "final_graphs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. CLINICAL FEATURE IMPORTANCE
# ============================================================

clinical_df = pd.read_csv(CLINICAL_FILE)

clinical_df = clinical_df.sort_values(
    "importance",
    ascending=True
)

# Show top 10
clinical_top = clinical_df.tail(10)

plt.figure(figsize=(9, 6))

plt.barh(
    clinical_top["feature"],
    clinical_top["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Clinical Feature")
plt.title("Top Clinical Features - Random Forest")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "clinical_feature_importance.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. IMAGE FEATURE IMPORTANCE
# ============================================================

image_df = pd.read_csv(IMAGE_FILE)

image_df = image_df.sort_values(
    "importance",
    ascending=True
)

# Show all 13 features
plt.figure(figsize=(9, 7))

plt.barh(
    image_df["feature"],
    image_df["importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Image Feature")
plt.title("Ultrasound Image Features - Random Forest")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "image_feature_importance.png"
    ),
    dpi=300
)

plt.close()


print("=" * 60)
print("FEATURE IMPORTANCE GRAPHS GENERATED")
print("=" * 60)

print()
print("Clinical graph:")
print(
    os.path.join(
        OUTPUT_DIR,
        "clinical_feature_importance.png"
    )
)

print()
print("Image graph:")
print(
    os.path.join(
        OUTPUT_DIR,
        "image_feature_importance.png"
    )
)