import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

DATA_PATH = BASE_DIR + r"\dataset\clinical\processed_ckd_dataset.csv"
TEST_PATH = BASE_DIR + r"\dataset\clinical\clinical_test.csv"

df = pd.read_csv(DATA_PATH)

# Keep the test set separate
_, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["ckd_pred"]
)

test_df.to_csv(TEST_PATH, index=False)

print("Total records:", len(df))
print("Test records:", len(test_df))
print("\nTest class distribution:")
print(test_df["ckd_pred"].value_counts())

print("\nSaved to:")
print(TEST_PATH)