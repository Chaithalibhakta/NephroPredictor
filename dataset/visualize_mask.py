import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==============================
# FILE PATHS
# ==============================

image_path = "kidney_images/images/1.jpg"
mask_path = "kidney_images/masks/1.mat"


# ==============================
# LOAD IMAGE
# ==============================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# OpenCV loads BGR → convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==============================
# LOAD MASK
# ==============================

mat_data = sio.loadmat(mask_path)

mask = mat_data["mask"]

print("Image shape:", image.shape)
print("Mask shape :", mask.shape)

print("Mask values:", np.unique(mask))


# ==============================
# RESIZE MASK IF NECESSARY
# ==============================

if image.shape[:2] != mask.shape:

    print("Resizing mask to match image...")

    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )


# ==============================
# CREATE OVERLAY
# ==============================

overlay = image.copy()

# Create mask region
mask_region = mask == 1

# Highlight mask region
overlay[mask_region] = [255, 0, 0]

# Blend original + mask
blended = cv2.addWeighted(
    image,
    0.7,
    overlay,
    0.3,
    0
)


# ==============================
# DISPLAY
# ==============================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Ultrasound")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Binary Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(blended)
plt.title("Mask Overlay")
plt.axis("off")

plt.tight_layout()

plt.show()