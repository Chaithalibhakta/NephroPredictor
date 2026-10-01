import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from unet_model import UNet


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

SEGMENTATION_DIR = os.path.join(
    BASE_DIR,
    "segmentation"
)

IMAGE_DIR = os.path.join(
    SEGMENTATION_DIR,
    "images"
)

MASK_DIR = os.path.join(
    SEGMENTATION_DIR,
    "masks"
)

CSV_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "kidney_images"
)

TEST_CSV = os.path.join(
    CSV_DIR,
    "test.csv"
)

MODEL_DIR = os.path.join(
    SEGMENTATION_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_unet.pth"
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 1
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FINAL U-NET TEST EVALUATION")
print("=" * 60)

print("\nDevice:", device)


# ============================================================
# DATASET CLASS
# ============================================================

class KidneyTestDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        mask_dir
    ):

        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):

        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        # ----------------------------------------------------
        # Get image filename from CSV
        # ----------------------------------------------------

        image_name = str(
            row["image_path"]
        )

        # Remove directory information
        image_name = os.path.basename(
            image_name
        )

        # Remove extension
        image_id = os.path.splitext(
            image_name
        )[0]

        # ----------------------------------------------------
        # Construct actual paths
        # ----------------------------------------------------

        image_path = os.path.join(
            self.image_dir,
            image_id + ".jpg"
        )

        mask_path = os.path.join(
            self.mask_dir,
            image_id + ".png"
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        # BGR -> RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Normalize
        image = image.astype(
            np.float32
        ) / 255.0

        # HWC -> CHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask = cv2.imread(
            mask_path,
            cv2.IMREAD_GRAYSCALE
        )

        if mask is None:

            raise RuntimeError(
                f"Could not read mask: {mask_path}"
            )

        # Normalize mask
        mask = mask.astype(
            np.float32
        ) / 255.0

        # Binary mask
        mask = (
            mask > 0.5
        ).astype(
            np.float32
        )

        # Add channel
        mask = np.expand_dims(
            mask,
            axis=0
        )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.tensor(
            image,
            dtype=torch.float32
        )

        mask = torch.tensor(
            mask,
            dtype=torch.float32
        )

        return image, mask


# ============================================================
# DICE SCORE
# ============================================================

def dice_score(
    prediction,
    target
):

    prediction = torch.sigmoid(
        prediction
    )

    prediction = (
        prediction > 0.5
    ).float()

    smooth = 1e-6

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return dice.item()


# ============================================================
# IOU SCORE
# ============================================================

def iou_score(
    prediction,
    target
):

    prediction = torch.sigmoid(
        prediction
    )

    prediction = (
        prediction > 0.5
    ).float()

    smooth = 1e-6

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()


# ============================================================
# LOSS FUNCTIONS
# ============================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(
    prediction,
    target
):

    prediction = torch.sigmoid(
        prediction
    )

    smooth = 1e-6

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return 1 - dice


def combined_loss(
    prediction,
    target
):

    bce = bce_loss(
        prediction,
        target
    )

    dice = dice_loss(
        prediction,
        target
    )

    return bce + dice


# ============================================================
# LOAD TEST CSV
# ============================================================

print("\nLoading test dataset...")

test_df = pd.read_csv(
    TEST_CSV
)

print(
    "Test samples:",
    len(test_df)
)


# ============================================================
# CREATE TEST DATASET
# ============================================================

test_dataset = KidneyTestDataset(
    test_df,
    IMAGE_DIR,
    MASK_DIR
)


# ============================================================
# CREATE TEST DATALOADER
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# CREATE U-NET
# ============================================================

print("\nCreating U-Net...")

model = UNet().to(device)

print(
    "U-Net created successfully."
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

print(
    "✓ Best U-Net model loaded successfully."
)


# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TEST EVALUATION")
print("=" * 60)


total_test_loss = 0.0
total_dice = 0.0
total_iou = 0.0


with torch.no_grad():

    for batch_index, (
        images,
        masks
    ) in enumerate(test_loader):

        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(
            images
        )

        # Loss
        loss = combined_loss(
            outputs,
            masks
        )

        # Metrics
        dice = dice_score(
            outputs,
            masks
        )

        iou = iou_score(
            outputs,
            masks
        )

        total_test_loss += loss.item()
        total_dice += dice
        total_iou += iou

        # Progress
        if (
            (batch_index + 1) % 100 == 0
        ):

            print(
                f"Test Batch "
                f"{batch_index + 1}/"
                f"{len(test_loader)}"
            )


# ============================================================
# FINAL RESULTS
# ============================================================

test_loss = (
    total_test_loss
    / len(test_loader)
)

test_dice = (
    total_dice
    / len(test_loader)
)

test_iou = (
    total_iou
    / len(test_loader)
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("FINAL U-NET TEST RESULTS")
print("=" * 60)

print(
    f"Test Loss : {test_loss:.4f}"
)

print(
    f"Test Dice : {test_dice:.4f}"
)

print(
    f"Test IoU  : {test_iou:.4f}"
)

print("=" * 60)

print(
    "\nModel:",
    MODEL_PATH
)

print(
    "\n✓ TEST EVALUATION COMPLETE"
)