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

TRAIN_CSV = os.path.join(
    CSV_DIR,
    "train.csv"
)

VAL_CSV = os.path.join(
    CSV_DIR,
    "val.csv"
)

MODEL_DIR = os.path.join(
    SEGMENTATION_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 1

EPOCHS = 5

LEARNING_RATE = 0.001

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("FINAL U-NET TRAINING")
print("=" * 60)

print("\nDevice:", device)

if device.type == "cpu":
    print("CPU training enabled.")
else:
    print("GPU training enabled.")


# ============================================================
# DATASET CLASS
# ============================================================

class KidneyDataset(Dataset):

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
        # Get image filename
        # ----------------------------------------------------

        image_name = str(
            row["image_path"]
        )

        # ----------------------------------------------------
        # Remove possible path information
        # ----------------------------------------------------

        image_name = os.path.basename(
            image_name
        )

        image_id = os.path.splitext(
            image_name
        )[0]

        # ----------------------------------------------------
        # Paths
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

        # BGR → RGB

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Normalize

        image = image.astype(
            np.float32
        ) / 255.0

        # HWC → CHW

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
# LOAD CSV FILES
# ============================================================

print("\nLoading dataset CSV files...")

train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

print(
    "Training samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = KidneyDataset(
    train_df,
    IMAGE_DIR,
    MASK_DIR
)

val_dataset = KidneyDataset(
    val_df,
    IMAGE_DIR,
    MASK_DIR
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating U-Net...")

model = UNet().to(device)

print("U-Net created successfully.")


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
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

best_model_path = os.path.join(
    MODEL_DIR,
    "best_unet.pth"
)


print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    # ========================================================
    # TRAINING
    # ========================================================

    model.train()

    total_train_loss = 0.0

    for batch_index, (
        images,
        masks
    ) in enumerate(train_loader):

        images = images.to(device)

        masks = masks.to(device)

        # Clear gradients

        optimizer.zero_grad()

        # Forward pass

        outputs = model(images)

        # Calculate loss

        loss = combined_loss(
            outputs,
            masks
        )

        # Backpropagation

        loss.backward()

        # Update weights

        optimizer.step()

        total_train_loss += loss.item()

        # Progress every 100 batches

        if (
            (batch_index + 1) % 100 == 0
        ):

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Batch "
                f"{batch_index + 1}/"
                f"{len(train_loader)}"
            )

    train_loss = (
        total_train_loss
        / len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0.0

    total_dice = 0.0

    total_iou = 0.0

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)

            masks = masks.to(device)

            outputs = model(images)

            loss = combined_loss(
                outputs,
                masks
            )

            total_val_loss += loss.item()

            total_dice += dice_score(
                outputs,
                masks
            )

            total_iou += iou_score(
                outputs,
                masks
            )


    val_loss = (
        total_val_loss
        / len(val_loader)
    )

    val_dice = (
        total_dice
        / len(val_loader)
    )

    val_iou = (
        total_iou
        / len(val_loader)
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("-" * 60)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print(
        f"Dice Score      : {val_dice:.4f}"
    )

    print(
        f"IoU Score       : {val_iou:.4f}"
    )

    print("-" * 60)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "✓ Best U-Net model saved."
        )

    print("\n")


# ============================================================
# COMPLETE
# ============================================================

print("=" * 60)
print("FINAL U-NET TRAINING COMPLETE")
print("=" * 60)

print(
    "\nBest validation loss:",
    best_val_loss
)

print(
    "\nBest model:",
    best_model_path
)