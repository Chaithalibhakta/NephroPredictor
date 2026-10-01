import os
import re
import pandas as pd

# ==============================
# PATHS
# ==============================

BASE_DIR = "kidney_images"

IMAGE_DIR = os.path.join(BASE_DIR, "images")
LABEL_DIR = os.path.join(BASE_DIR, "labels")
MASK_DIR = os.path.join(BASE_DIR, "masks")

OUTPUT_CSV = os.path.join(BASE_DIR, "kidney_dataset.csv")


# ==============================
# READ ALL LABEL FILES
# ==============================

records = []

label_files = sorted(
    [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")],
    key=lambda x: int(os.path.splitext(x)[0])
)

print(f"Found {len(label_files)} label files.")


for label_file in label_files:

    label_path = os.path.join(LABEL_DIR, label_file)

    # Image ID
    image_id = os.path.splitext(label_file)[0]

    image_filename = f"{image_id}.jpg"
    image_path = os.path.join(IMAGE_DIR, image_filename)

    mask_filename = f"{image_id}.mat"
    mask_path = os.path.join(MASK_DIR, mask_filename)

    # Default values
    diagnosis = None
    pathology_1 = None
    pathology_2 = None
    view = None

    bounding_boxes = []

    # ==============================
    # READ TXT FILE
    # ==============================

    with open(label_path, "r", encoding="utf-8") as f:

        lines = f.readlines()

    for line in lines:

        line = line.strip()

        # Global diagnosis
        if line.startswith("Global diagnosis:"):
            diagnosis = int(
                line.split(":")[1].strip()
            )

        # Global pathologies
        elif line.startswith("Global pathologies:"):

            values = line.split(":")[1].strip().split()

            if len(values) >= 2:
                pathology_1 = int(values[0])
                pathology_2 = int(values[1])

        # View
        elif line.startswith("View:"):
            view = int(
                line.split(":")[1].strip()
            )

        # Bounding boxes
        elif line.startswith("Bounding box"):

            values = line.split(":")[1].strip().split()

            if len(values) >= 5:

                x = int(values[0])
                y = int(values[1])
                width = int(values[2])
                height = int(values[3])
                class_id = int(values[4])

                bounding_boxes.append(
                    [x, y, width, height, class_id]
                )

    # ==============================
    # FILE EXISTENCE CHECK
    # ==============================

    image_exists = os.path.exists(image_path)
    mask_exists = os.path.exists(mask_path)

    # ==============================
    # STORE RECORD
    # ==============================

    records.append({

        "image_id": image_id,

        "image_path": image_path,

        "label_path": label_path,

        "mask_path": mask_path,

        "image_exists": image_exists,

        "mask_exists": mask_exists,

        "diagnosis": diagnosis,

        "pathology_1": pathology_1,

        "pathology_2": pathology_2,

        "view": view,

        "num_bounding_boxes": len(bounding_boxes),

        "bounding_boxes": str(bounding_boxes)

    })


# ==============================
# CREATE DATAFRAME
# ==============================

df = pd.DataFrame(records)


# ==============================
# SAVE CSV
# ==============================

df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ==============================
# SUMMARY
# ==============================

print("\n================================")
print("DATASET CREATION COMPLETE")
print("================================")

print(f"Total records: {len(df)}")

print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts(dropna=False))

print("\nMissing diagnosis:")
print(df["diagnosis"].isna().sum())

print("\nMissing images:")
print((~df["image_exists"]).sum())

print("\nMissing masks:")
print((~df["mask_exists"]).sum())

print("\nCSV saved to:")
print(OUTPUT_CSV)