import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("../dataset/clinical/cleaned_ckd_dataset.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)

# -----------------------------
# Remove unwanted column
# -----------------------------
if "cluster" in df.columns:
    df = df.drop(columns=["cluster"])
    print("Cluster column removed.")

# -----------------------------
# Separate categorical & numerical columns
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# -----------------------------
# Handle Missing Values
# -----------------------------
num_imputer = SimpleImputer(strategy="median")
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Missing values handled successfully!")

# -----------------------------
# Encode Categorical Features
# -----------------------------
label_encoders = {}

for col in categorical_cols:
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    label_encoders[col] = encoder

print("Categorical encoding completed!")

# -----------------------------
# Scale Numerical Features
# -----------------------------
feature_columns = [
    col for col in df.columns
    if col not in ["ckd_pred", "ckd_stage"]
]

scaler = StandardScaler()

df[feature_columns] = scaler.fit_transform(df[feature_columns])

print("Feature scaling completed!")

# -----------------------------
# Save Processed Dataset
# -----------------------------
df.to_csv(
    "../dataset/clinical/processed_ckd_dataset.csv",
    index=False
)

# -----------------------------
# Save Scaler & Encoders
# -----------------------------
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(label_encoders, "../models/label_encoders.pkl")

print("Processed dataset saved successfully!")
print("Scaler saved successfully!")
print("Label Encoders saved successfully!")

print("\nFinal Features Used For Training:\n")
print(feature_columns)