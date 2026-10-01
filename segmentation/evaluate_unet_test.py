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

SEG_DIR = os.path.join(
    BASE_DIR,
    "segmentation"
)

IMAGE_DIR = os.path.join(
    SEG_DIR,
    "images"
)

MASK_DIR = os.path.join(
    SEG_DIR,
    "masks"
)

CSV_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "kidney_images"
)

MODEL_PATH = os.path.join(
    SEG_DIR,
    "models",
    "best_unet.pth"
)

OUTPUT_DIR = os.path.join(
    SEG_DIR,
    "features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 256
THRESHOLD = 0.5

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. LOAD U-NET
# ============================================================

print("\nLoading U-Net model...")

model = UNet().to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    state_dict
)

model.eval()

print("U-Net loaded successfully.")


# ============================================================
# 4. METRIC FUNCTIONS
# ============================================================

def calculate_dice(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        np.uint8
    )

    ground_truth = ground_truth.astype(
        np.uint8
    )

    intersection = np.sum(
        prediction * ground_truth
    )

    prediction_sum = np.sum(
        prediction
    )

    ground_truth_sum = np.sum(
        ground_truth
    )

    denominator = (
        prediction_sum +
        ground_truth_sum
    )

    if denominator == 0:

        if prediction_sum == 0 and ground_truth_sum == 0:
            return 1.0

        return 0.0

    dice = (
        2.0 * intersection
    ) / denominator

    return float(dice)


def calculate_iou(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        np.uint8
    )

    ground_truth = ground_truth.astype(
        np.uint8
    )

    intersection = np.sum(
        prediction * ground_truth
    )

    union = (
        np.sum(prediction) +
        np.sum(ground_truth) -
        intersection
    )

    if union == 0:

        if (
            np.sum(prediction) == 0
            and
            np.sum(ground_truth) == 0
        ):
            return 1.0

        return 0.0

    iou = (
        intersection / union
    )

    return float(iou)


def calculate_precision(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        np.uint8
    )

    ground_truth = ground_truth.astype(
        np.uint8
    )

    true_positive = np.sum(
        (prediction == 1) &
        (ground_truth == 1)
    )

    false_positive = np.sum(
        (prediction == 1) &
        (ground_truth == 0)
    )

    denominator = (
        true_positive +
        false_positive
    )

    if denominator == 0:

        return 1.0 if np.sum(ground_truth) == 0 else 0.0

    return float(
        true_positive / denominator
    )


def calculate_recall(
    prediction,
    ground_truth
):

    prediction = prediction.astype(
        np.uint8
    )

    ground_truth = ground_truth.astype(
        np.uint8
    )

    true_positive = np.sum(
        (prediction == 1) &
        (ground_truth == 1)
    )

    false_negative = np.sum(
        (prediction == 0) &
        (ground_truth == 1)
    )

    denominator = (
        true_positive +
        false_negative
    )

    if denominator == 0:

        return 1.0 if np.sum(prediction) == 0 else 0.0

    return float(
        true_positive / denominator
    )


# ============================================================
# 5. LOAD TEST CSV
# ============================================================

TEST_CSV = os.path.join(
    CSV_DIR,
    "test.csv"
)

test_df = pd.read_csv(
    TEST_CSV
)

print(
    "\nTest images:",
    len(test_df)
)


# ============================================================
# 6. EVALUATION
# ============================================================

results = []


print("\n")
print("=" * 70)
print("STARTING U-NET TEST EVALUATION")
print("=" * 70)


for index, row in test_df.iterrows():

    image_id = str(
        row["image_id"]
    )

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".jpg"
    )

    mask_path = os.path.join(
        MASK_DIR,
        image_id + ".png"
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        image_path
    ):

        print(
            "WARNING: Image not found:",
            image_path
        )

        continue


    if not os.path.exists(
        mask_path
    ):

        print(
            "WARNING: Mask not found:",
            mask_path
        )

        continue


    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        print(
            "ERROR reading image:",
            image_path
        )

        continue


    # --------------------------------------------------------
    # Convert BGR -> RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Resize image
    # --------------------------------------------------------

    image_resized = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    image_float = (
        image_resized.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # HWC -> CHW
    # --------------------------------------------------------

    image_chw = np.transpose(
        image_float,
        (2, 0, 1)
    )


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    image_tensor = torch.tensor(
        image_chw,
        dtype=torch.float32
    ).unsqueeze(0).to(device)


    # ========================================================
    # U-NET PREDICTION
    # ========================================================

    with torch.no_grad():

        output = model(
            image_tensor
        )

        output = torch.sigmoid(
            output
        )

        prediction = (
            output > THRESHOLD
        ).float()


    # --------------------------------------------------------
    # Convert prediction to NumPy
    # --------------------------------------------------------

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )


    # ========================================================
    # LOAD GROUND-TRUTH MASK
    # ========================================================

    ground_truth = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if ground_truth is None:

        print(
            "ERROR reading mask:",
            mask_path
        )

        continue


    # --------------------------------------------------------
    # Resize mask
    # --------------------------------------------------------

    ground_truth = cv2.resize(
        ground_truth,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # --------------------------------------------------------
    # Convert to binary
    # --------------------------------------------------------

    ground_truth = (
        ground_truth > 127
    ).astype(
        np.uint8
    )


    # ========================================================
    # CALCULATE METRICS
    # ========================================================

    dice = calculate_dice(
        prediction,
        ground_truth
    )

    iou = calculate_iou(
        prediction,
        ground_truth
    )

    precision = calculate_precision(
        prediction,
        ground_truth
    )

    recall = calculate_recall(
        prediction,
        ground_truth
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "image_id":
            image_id,

        "diagnosis":
            int(row["diagnosis"]),

        "dice":
            dice,

        "iou":
            iou,

        "precision":
            precision,

        "recall":
            recall
    })


    print(
        f"Processed "
        f"{index + 1}/{len(test_df)}"
        f" | ID: {image_id}"
        f" | Dice: {dice:.4f}"
        f" | IoU: {iou:.4f}"
    )


# ============================================================
# 7. CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 8. SAVE INDIVIDUAL RESULTS
# ============================================================

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "unet_test_image_metrics.csv"
)

results_df.to_csv(
    RESULT_FILE,
    index=False
)


# ============================================================
# 9. CALCULATE OVERALL METRICS
# ============================================================

mean_dice = results_df[
    "dice"
].mean()

mean_iou = results_df[
    "iou"
].mean()

mean_precision = results_df[
    "precision"
].mean()

mean_recall = results_df[
    "recall"
].mean()


# ============================================================
# 10. SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame({

    "Metric": [
        "Dice",
        "IoU",
        "Precision",
        "Recall"
    ],

    "Value": [
        mean_dice,
        mean_iou,
        mean_precision,
        mean_recall
    ]
})


SUMMARY_FILE = os.path.join(
    OUTPUT_DIR,
    "unet_test_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# 11. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("U-NET TEST EVALUATION COMPLETED")
print("=" * 70)

print(
    f"\nTest images evaluated: "
    f"{len(results_df)}"
)

print(
    f"\nMean Dice Score: "
    f"{mean_dice:.4f}"
)

print(
    f"Mean IoU: "
    f"{mean_iou:.4f}"
)

print(
    f"Mean Precision: "
    f"{mean_precision:.4f}"
)

print(
    f"Mean Recall: "
    f"{mean_recall:.4f}"
)

print("\nSaved files:")

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)

print("=" * 70)