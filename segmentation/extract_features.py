import os
import cv2
import numpy as np
import pandas as pd
import torch

from unet_model import UNet


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

SEG_DIR = os.path.join(BASE_DIR, "segmentation")

IMAGE_DIR = os.path.join(SEG_DIR, "images")
MASK_DIR = os.path.join(SEG_DIR, "masks")

CSV_DIR = os.path.join(BASE_DIR, "dataset", "kidney_images")

MODEL_PATH = os.path.join(
    SEG_DIR,
    "models",
    "best_unet.pth"
)

OUTPUT_DIR = os.path.join(
    SEG_DIR,
    "features"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

TEST_MODE = False
TEST_SAMPLES = 5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. LOAD TRAINED U-NET
# ============================================================

print("\nLoading U-Net model...")

model = UNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(state_dict)

model.eval()

print("U-Net loaded successfully.")


# ============================================================
# 4. FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(image_path):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        print(
            "ERROR: Could not read image:",
            image_path
        )
        return None

    print(
        "Image:",
        os.path.basename(image_path),
        "| Shape:",
        image.shape
    )


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Prepare image for U-Net
    # --------------------------------------------------------

    image_float = (
        rgb_image.astype(np.float32) / 255.0
    )

    image_chw = np.transpose(
        image_float,
        (2, 0, 1)
    )

    image_tensor = torch.tensor(
        image_chw,
        dtype=torch.float32
    ).unsqueeze(0).to(device)


    # --------------------------------------------------------
    # U-Net prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = model(
            image_tensor
        )

        prediction = torch.sigmoid(
            prediction
        )

        predicted_mask = (
            prediction > 0.5
        ).float()


    # --------------------------------------------------------
    # Convert prediction to NumPy
    # --------------------------------------------------------

    binary_mask = (
        predicted_mask
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    binary_mask = binary_mask * 255

    print(
        "Predicted mask shape:",
        binary_mask.shape
    )


    # ========================================================
    # 5. KIDNEY AREA
    # ========================================================

    kidney_area = cv2.countNonZero(
        binary_mask
    )


    # ========================================================
    # 6. CONTOURS
    # ========================================================

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # Default values
    # Used when no contour exists

    kidney_perimeter = 0.0
    kidney_width = 0.0
    kidney_height = 0.0
    aspect_ratio = 0.0
    circularity = 0.0


    # ========================================================
    # 7. LARGEST KIDNEY CONTOUR
    # ========================================================

    if len(contours) > 0:

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        contour_area = cv2.contourArea(
            largest_contour
        )

        kidney_perimeter = cv2.arcLength(
            largest_contour,
            True
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        kidney_width = float(w)

        kidney_height = float(h)


        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

        if kidney_height > 0:

            aspect_ratio = (
                kidney_width /
                kidney_height
            )


        # ----------------------------------------------------
        # Circularity
        # ----------------------------------------------------

        if kidney_perimeter > 0:

            circularity = (
                4 * np.pi * contour_area
            ) / (
                kidney_perimeter ** 2
            )


    # ========================================================
    # 8. IMAGE INTENSITY FEATURES
    # ========================================================

    grayscale = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    mean_intensity = float(
        np.mean(grayscale)
    )

    std_intensity = float(
        np.std(grayscale)
    )

    min_intensity = float(
        np.min(grayscale)
    )

    max_intensity = float(
        np.max(grayscale)
    )

    median_intensity = float(
        np.median(grayscale)
    )


    # ========================================================
    # 9. ROI FEATURES
    # ========================================================

    roi_mean = 0.0
    roi_std = 0.0


    if len(contours) > 0:

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        if w > 0 and h > 0:

            roi = grayscale[
                y:y + h,
                x:x + w
            ]

            if roi.size > 0:

                roi_mean = float(
                    np.mean(roi)
                )

                roi_std = float(
                    np.std(roi)
                )


    # ========================================================
    # 10. RETURN FEATURES
    # ========================================================

    features = {

        "kidney_area":
            float(kidney_area),

        "kidney_perimeter":
            float(kidney_perimeter),

        "kidney_width":
            float(kidney_width),

        "kidney_height":
            float(kidney_height),

        "aspect_ratio":
            float(aspect_ratio),

        "circularity":
            float(circularity),

        "mean_intensity":
            mean_intensity,

        "std_intensity":
            std_intensity,

        "min_intensity":
            min_intensity,

        "max_intensity":
            max_intensity,

        "median_intensity":
            median_intensity,

        "roi_mean":
            roi_mean,

        "roi_std":
            roi_std
    }

    return features


# ============================================================
# 11. PROCESS DATASET
# ============================================================

def process_dataset(
    csv_filename,
    output_filename
):

    print("\n")
    print("=" * 70)
    print("PROCESSING:", csv_filename)
    print("=" * 70)


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        CSV_DIR,
        csv_filename
    )

    df = pd.read_csv(
        csv_path
    )

    print(
        "Total records:",
        len(df)
    )


    # --------------------------------------------------------
    # Test mode
    # --------------------------------------------------------

    if TEST_MODE:

        df = df.head(
            TEST_SAMPLES
        )

        print(
            "TEST MODE:",
            len(df),
            "images"
        )


    results = []


    # ========================================================
    # PROCESS EACH IMAGE
    # ========================================================

    for index, row in df.iterrows():

        try:

            # ------------------------------------------------
            # Get image name from CSV
            # ------------------------------------------------

            image_name = os.path.basename(
                str(row["image_path"])
            )

            image_id = os.path.splitext(
                image_name
            )[0]


            # ------------------------------------------------
            # Build actual image path
            # ------------------------------------------------

            image_path = os.path.join(
                IMAGE_DIR,
                image_id + ".jpg"
            )


            # ------------------------------------------------
            # Check image exists
            # ------------------------------------------------

            if not os.path.exists(
                image_path
            ):

                print(
                    "WARNING: Image not found:",
                    image_path
                )

                continue


            # ------------------------------------------------
            # Extract image features
            # ------------------------------------------------

            features = extract_features(
                image_path
            )


            if features is None:
                continue


            # ------------------------------------------------
            # Add metadata
            # ------------------------------------------------

            features["image_id"] = image_id

            features["diagnosis"] = int(
                row["diagnosis"]
            )

            features["class_name"] = (
                "healthy"
                if int(row["diagnosis"]) == 0
                else "pathological"
            )

            features["image_path"] = str(
                row["image_path"]
            )


            # ------------------------------------------------
            # Add result
            # ------------------------------------------------

            results.append(
                features
            )


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            print(
                f"Processed "
                f"{index + 1}/{len(df)}"
            )


        except Exception as e:

            print(
                "ERROR processing:",
                row.get(
                    "image_path",
                    "unknown"
                )
            )

            print(
                "Error:",
                e
            )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        results
    )


    # ========================================================
    # SAVE CSV
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    result_df.to_csv(
        output_path,
        index=False
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 70)
    print("DATASET COMPLETED")
    print("=" * 70)

    print(
        "Input CSV:",
        csv_filename
    )

    print(
        "Images processed:",
        len(result_df)
    )

    print(
        "Output file:",
        output_path
    )

    print(
        "Columns:",
        len(result_df.columns)
    )

    print("\nFeature columns:")

    for column in result_df.columns:

        print(
            " -",
            column
        )

    print("=" * 70)


# ============================================================
# 12. PROCESS TRAIN DATASET
# ============================================================

process_dataset(
    "train.csv",
    "train_image_features.csv"
)


# ============================================================
# 13. PROCESS VALIDATION DATASET
# ============================================================

process_dataset(
    "val.csv",
    "val_image_features.csv"
)


# ============================================================
# 14. PROCESS TEST DATASET
# ============================================================

process_dataset(
    "test.csv",
    "test_image_features.csv"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ALL FEATURE EXTRACTION COMPLETED")
print("=" * 70)

print(
    "Train features:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "train_image_features.csv"
    )
)

print(
    "\nValidation features:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "val_image_features.csv"
    )
)

print(
    "\nTest features:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "test_image_features.csv"
    )
)

print("=" * 70)

