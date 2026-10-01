import os
import cv2
import numpy as np
import pandas as pd
import torch
import joblib

from unet_model import UNet

# ======================================================
# PATHS
# ======================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

IMAGE_DIR = os.path.join(BASE_DIR, "segmentation", "images")
MODEL_PATH = os.path.join(
    BASE_DIR, "segmentation", "models", "best_unet.pth"
)

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "image_random_forest.pkl"
)

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

# ======================================================
# LOAD MODELS
# ======================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

print("\nLoading U-Net...")
unet = UNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

unet.load_state_dict(state_dict)
unet.eval()

print("U-Net loaded.")

print("\nLoading image classifier...")
image_model = joblib.load(IMAGE_MODEL_PATH)

print("Image Random Forest loaded.")

# ======================================================
# SELECT ONE IMAGE
# ======================================================

images = [
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if len(images) == 0:
    raise FileNotFoundError("No images found.")

image_name = images[0]

image_path = os.path.join(
    IMAGE_DIR,
    image_name
)

print("\nTesting image:")
print(image_path)

# ======================================================
# READ IMAGE
# ======================================================

image = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)

if image is None:
    raise ValueError("Could not read image.")

original = image.copy()

# Resize for U-Net
resized = cv2.resize(
    image,
    (256, 256)
)

rgb = cv2.cvtColor(
    resized,
    cv2.COLOR_BGR2RGB
)

normalized = rgb.astype(np.float32) / 255.0

tensor = torch.from_numpy(
    normalized.transpose(2, 0, 1)
).unsqueeze(0).to(device)

# ======================================================
# U-NET PREDICTION
# ======================================================

with torch.no_grad():

    output = unet(tensor)

    probability = torch.sigmoid(output)

    predicted_mask = (
        probability > 0.5
    ).float()

mask = predicted_mask[0, 0].cpu().numpy().astype(np.uint8)

# ======================================================
# EXTRACT IMAGE FEATURES
# ======================================================

mask_uint8 = mask * 255

contours, _ = cv2.findContours(
    mask_uint8,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:

    print("\nWARNING: No kidney region detected.")

    features = {
        "kidney_area": 0,
        "kidney_perimeter": 0,
        "kidney_width": 0,
        "kidney_height": 0,
        "aspect_ratio": 0,
        "circularity": 0,
        "mean_intensity": 0,
        "std_intensity": 0,
        "min_intensity": 0,
        "max_intensity": 0,
        "median_intensity": 0,
        "roi_mean": 0,
        "roi_std": 0
    }

else:

    contour = max(
        contours,
        key=cv2.contourArea
    )

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(
        contour,
        True
    )

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = (
        w / h if h != 0 else 0
    )

    circularity = (
        (4 * np.pi * area) /
        (perimeter ** 2)
        if perimeter != 0
        else 0
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    mask_bool = mask.astype(bool)

    pixels = gray[mask_bool]

    if len(pixels) == 0:

        mean_intensity = 0
        std_intensity = 0
        min_intensity = 0
        max_intensity = 0
        median_intensity = 0

    else:

        mean_intensity = float(np.mean(pixels))
        std_intensity = float(np.std(pixels))
        min_intensity = float(np.min(pixels))
        max_intensity = float(np.max(pixels))
        median_intensity = float(np.median(pixels))

    roi = gray[y:y+h, x:x+w]

    roi_mean = float(np.mean(roi))
    roi_std = float(np.std(roi))

    features = {
        "kidney_area": area,
        "kidney_perimeter": perimeter,
        "kidney_width": w,
        "kidney_height": h,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "median_intensity": median_intensity,
        "roi_mean": roi_mean,
        "roi_std": roi_std
    }

# ======================================================
# CLASSIFICATION
# ======================================================

feature_df = pd.DataFrame(
    [[features[col] for col in FEATURE_COLUMNS]],
    columns=FEATURE_COLUMNS
)

prediction = int(
    image_model.predict(feature_df)[0]
)

probabilities = image_model.predict_proba(
    feature_df
)[0]

# ======================================================
# OUTPUT
# ======================================================

print("\n" + "=" * 60)
print("IMAGE PIPELINE RESULT")
print("=" * 60)

print("\nImage:", image_name)

print("\nExtracted Features:")
print(feature_df.to_string(index=False))

print("\nPrediction:")

if prediction == 1:
    print("Pathological")
else:
    print("Healthy")

print("\nProbabilities:")
print("Healthy:", round(float(probabilities[0]) * 100, 2), "%")
print("Pathological:", round(float(probabilities[1]) * 100, 2), "%")

print("\n" + "=" * 60)
print("IMAGE PIPELINE TEST COMPLETED")
print("=" * 60)