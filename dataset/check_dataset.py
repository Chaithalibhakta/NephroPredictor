import pandas as pd

# Load processed dataset
df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

print("=" * 80)
print("FIRST 5 ROWS")
print("=" * 80)
print(df.head())

print("\n")

print("=" * 80)
print("COLUMN NAMES")
print("=" * 80)
print(df.columns.tolist())

print("\n")

print("=" * 80)
print("DATA TYPES")
print("=" * 80)
print(df.dtypes)

print("\n")

print("=" * 80)
print("CKD LABEL COUNTS")
print("=" * 80)
print(df["ckd_pred"].value_counts())

print("\n")

print("=" * 80)
print("FIRST 5 ROWS (TRANSPOSED)")
print("=" * 80)
print(df.head().T)

import joblib

model = joblib.load("../models/best_ckd_model.pkl")

print(model)


import pandas as pd

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

print(df.columns.tolist())