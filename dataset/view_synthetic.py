import os
import cv2
import matplotlib.pyplot as plt

IMAGE_DIR = "kidney_images/synthetic_test/images"
MASK_DIR = "kidney_images/synthetic_test/masks"

files = sorted(os.listdir(IMAGE_DIR))[:6]

plt.figure(figsize=(12, 12))

for i, filename in enumerate(files):

    image_path = os.path.join(IMAGE_DIR, filename)

    mask_filename = os.path.splitext(filename)[0] + ".png"

    mask_path = os.path.join(MASK_DIR, mask_filename)

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

    blended = cv2.addWeighted(
        image,
        0.7,
        overlay,
        0.3,
        0
    )

    plt.subplot(3, 2, i + 1)

    plt.imshow(blended)

    plt.title(filename)

    plt.axis("off")

plt.tight_layout()

plt.show()