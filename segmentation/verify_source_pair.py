
import os
import cv2
import numpy as np
from scipy.io import loadmat

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

IMAGE_SOURCE = os.path.join(
    BASE_DIR, "dataset", "kidney_images", "images"
)

MASK_SOURCE = os.path.join(
    BASE_DIR, "dataset", "kidney_images", "masks"
)

PREPARED_IMAGES = os.path.join(
    BASE_DIR, "segmentation", "images"
)

PREPARED_MASKS = os.path.join(
    BASE_DIR, "segmentation", "masks"
)

IMAGE_IDS = ["967", "1771"]


def report_mask(name, mask):
    print(f"\n{name}")
    print("Shape:", mask.shape)
    print("Data type:", mask.dtype)
    print("Unique values:", np.unique(mask)[:30])
    print("Minimum:", mask.min())
    print("Maximum:", mask.max())

    foreground = mask > 0

    print("Foreground pixels:", int(foreground.sum()))

    ys, xs = np.where(foreground)

    if len(xs):
        print(
            "Foreground bounding box:",
            (int(xs.min()), int(ys.min()),
             int(xs.max()), int(ys.max()))
        )
    else:
        print("Foreground bounding box: None")


for image_id in IMAGE_IDS:
    print("\n" + "=" * 60)
    print("CHECKING IMAGE ID:", image_id)
    print("=" * 60)

    original_image_path = os.path.join(
        IMAGE_SOURCE, image_id + ".jpg"
    )

    original_mat_path = os.path.join(
        MASK_SOURCE, image_id + ".mat"
    )

    prepared_image_path = os.path.join(
        PREPARED_IMAGES, image_id + ".jpg"
    )

    prepared_mask_path = os.path.join(
        PREPARED_MASKS, image_id + ".png"
    )

    # Check that all four files exist
    for label, path in [
        ("Original image", original_image_path),
        ("Original MAT mask", original_mat_path),
        ("Prepared image", prepared_image_path),
        ("Prepared PNG mask", prepared_mask_path),
    ]:
        print(f"{label} exists:", os.path.exists(path))

    if not all(os.path.exists(p) for p in [
        original_image_path,
        original_mat_path,
        prepared_image_path,
        prepared_mask_path
    ]):
        print("Missing file. Skipping this image.")
        continue

    original_image = cv2.imread(original_image_path)
    prepared_image = cv2.imread(prepared_image_path)

    print("\nOriginal image shape:", original_image.shape)
    print("Prepared image shape:", prepared_image.shape)

    # Inspect the original MAT file structure
    mat_data = loadmat(original_mat_path)

    print("\nMAT-file keys:", [
        key for key in mat_data.keys()
        if not key.startswith("__")
    ])

    if "mask" not in mat_data:
        print("ERROR: No 'mask' key in MAT file.")
        continue

    original_mask = np.asarray(mat_data["mask"])

    report_mask("Original MAT mask", original_mask)

    # Reproduce the preparation script's conversion
    converted_mask = original_mask.astype(np.uint8) * 255

    # Reproduce the preparation script's resizing
    expected_mask = cv2.resize(
        converted_mask,
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )

    prepared_mask = cv2.imread(
        prepared_mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    report_mask("Expected resized mask", expected_mask)
    report_mask("Saved prepared mask", prepared_mask)

    print(
        "\nPrepared mask matches recreated mask:",
        np.array_equal(expected_mask, prepared_mask)
    )

    print(
        "Original mask dimensions match original image:",
        original_mask.shape == original_image.shape[:2]
    )

print("\n" + "=" * 60)
print("SOURCE PAIR VERIFICATION COMPLETE")
print("No source files or model weights were modified.")
print("=" * 60)

