import pandas as pd

BASE_DIR = r"C:\Users\HP\OneDrive\Desktop\NephroPredictorr"

DATA_PATH = BASE_DIR + r"\dataset\clinical\processed_ckd_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMNS ==========")
for i, col in enumerate(df.columns):
    print(i, ":", col)

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== TARGET COLUMNS ==========")
for col in df.columns:
    if "ckd" in col.lower() or "diagn" in col.lower() or "target" in col.lower():
        print(col, df[col].unique()[:20])