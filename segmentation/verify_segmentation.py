import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr\segmentation"

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MASK_DIR = os.path.join(BASE_DIR, "masks")

# Check these 6 samples
sample_ids = [1, 100, 500, 1000, 1500, 1985]

fig, axes = plt.subplots(
    len(sample_ids),
    3,
    figsize=(12, 20)
)

for row, image_id in enumerate(sample_ids):

    image_path = os.path.join(
        IMAGE_DIR,
        f"{image_id}.jpg"
    )

    mask_path = os.path.join(
        MASK_DIR,
        f"{image_id}.png"
    )

    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    # Create overlay
    overlay = image.copy()

    overlay[mask > 0] = [255, 0, 0]

    # Blend original + mask
    blended = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    axes[row, 0].imshow(image)
    axes[row, 0].set_title(
        f"{image_id}.jpg - Original"
    )

    axes[row, 1].imshow(
        mask,
        cmap="gray"
    )
    axes[row, 1].set_title(
        f"{image_id}.png - Ground Truth Mask"
    )

    axes[row, 2].imshow(blended)
    axes[row, 2].set_title(
        f"{image_id} - Overlay"
    )

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()

output_path = os.path.join(
    BASE_DIR,
    "segmentation_verification.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.show()

print("\nVerification image saved to:")
print(output_path)