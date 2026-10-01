import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# ==========================================
# FILES
# ==========================================

image_path = "kidney_images/images/1.jpg"
mask_path = "kidney_images/masks/1.mat"

# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(image_path)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ==========================================
# LOAD MASK
# ==========================================

mat = sio.loadmat(mask_path)

mask = mat["mask"]

# ==========================================
# CREATE KIDNEY-FOCUSED IMAGE
# ==========================================

# Keep only kidney pixels
kidney_image = image.copy()

kidney_image[mask == 0] = 0

# ==========================================
# RESIZE
# ==========================================

TARGET_SIZE = (224, 224)

original_resized = cv2.resize(
    image,
    TARGET_SIZE,
    interpolation=cv2.INTER_AREA
)

mask_resized = cv2.resize(
    mask,
    TARGET_SIZE,
    interpolation=cv2.INTER_NEAREST
)

kidney_resized = cv2.resize(
    kidney_image,
    TARGET_SIZE,
    interpolation=cv2.INTER_AREA
)

# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)

plt.imshow(original_resized)

plt.title("Original 224×224")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(mask_resized, cmap="gray")

plt.title("Kidney Mask")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(kidney_resized)

plt.title("Kidney-Focused Image")

plt.axis("off")


plt.tight_layout()

plt.show()