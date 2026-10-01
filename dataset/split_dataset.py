import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# PATH
# ==========================================

CSV_PATH = "kidney_images/kidney_dataset.csv"

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(CSV_PATH)

print("Total samples:", len(df))

# ==========================================
# FIRST SPLIT
# 70% TRAIN
# 30% TEMPORARY
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["diagnosis"]
)

# ==========================================
# SECOND SPLIT
# 15% VALIDATION
# 15% TEST
# ==========================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["diagnosis"]
)

# ==========================================
# RESET INDEX
# ==========================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# ==========================================
# SAVE SPLITS
# ==========================================

train_df.to_csv(
    "kidney_images/train.csv",
    index=False
)

val_df.to_csv(
    "kidney_images/val.csv",
    index=False
)

test_df.to_csv(
    "kidney_images/test.csv",
    index=False
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n==========================================")
print("DATASET SPLIT COMPLETE")
print("==========================================")

print("\nTrain samples:", len(train_df))
print("Validation samples:", len(val_df))
print("Test samples:", len(test_df))

print("\n------------------------------------------")
print("TRAIN CLASS DISTRIBUTION")
print("------------------------------------------")

print(train_df["class_name"].value_counts())

print("\nTrain percentages:")
print(
    train_df["class_name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n------------------------------------------")
print("VALIDATION CLASS DISTRIBUTION")
print("------------------------------------------")

print(val_df["class_name"].value_counts())

print("\nValidation percentages:")
print(
    val_df["class_name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\n------------------------------------------")
print("TEST CLASS DISTRIBUTION")
print("------------------------------------------")

print(test_df["class_name"].value_counts())

print("\nTest percentages:")
print(
    test_df["class_name"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nFiles created:")
print("kidney_images/train.csv")
print("kidney_images/val.csv")
print("kidney_images/test.csv")