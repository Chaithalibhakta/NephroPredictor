import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

# =============================================
# Load Dataset
# =============================================
df = pd.read_csv("../dataset/clinical/updated_ckd_dataset_with_stages.csv")

print("Dataset Shape:", df.shape)

# =============================================
# Remove Cluster Column
# =============================================
if "cluster" in df.columns:
    df.drop(columns=["cluster"], inplace=True)
    print("Cluster column removed.")

# =============================================
# Handle Missing Values
# =============================================
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Missing values handled successfully.")

# =============================================
# Encode ALL categorical columns
# =============================================
label_encoders = {}

for col in categorical_cols:

    encoder = LabelEncoder()

    df[col] = encoder.fit_transform(df[col].astype(str))

    label_encoders[col] = encoder

print("Categorical columns encoded successfully.")

# =============================================
# Display CKD Mapping
# =============================================
if "ckd_pred" in label_encoders:

    mapping = {}

    encoder = label_encoders["ckd_pred"]

    for cls, value in zip(encoder.classes_, encoder.transform(encoder.classes_)):
        mapping[cls] = int(value)

    print("\nCKD Mapping")
    print(mapping)

# =============================================
# Separate Features & Target
# =============================================
feature_columns = [
    col for col in df.columns
    if col not in ["ckd_pred", "ckd_stage"]
]

X = df[feature_columns]

y = df["ckd_pred"]

# =============================================
# Scale Features
# =============================================
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

processed_df = pd.DataFrame(
    X_scaled,
    columns=feature_columns
)

processed_df["ckd_pred"] = y.values
processed_df["ckd_stage"] = df["ckd_stage"].values
# =============================================
# Save Processed Dataset
# =============================================
processed_path = "../dataset/clinical/processed_ckd_dataset.csv"

processed_df.to_csv(
    processed_path,
    index=False
)

print("Processed dataset saved.")

# =============================================
# Save Scaler
# =============================================
joblib.dump(
    scaler,
    "../models/scaler.pkl"
)

print("Scaler saved.")

# =============================================
# Save Label Encoders
# =============================================
joblib.dump(
    label_encoders,
    "../models/label_encoders.pkl"
)

print("Label Encoders saved.")

# =============================================
# Display Final Information
# =============================================
print("\nProcessed Dataset Shape:", processed_df.shape)

print("\nFeature Columns Used:")

for feature in feature_columns:
    print(feature)

print("\nFirst 5 Rows of Processed Dataset")
print(processed_df.head())

print("\nChecking Missing Values")
print(processed_df.isnull().sum())

print("\nProcessed Dataset Information")
print(processed_df.info())

print("\nScaler Mean")
print(scaler.mean_)

print("\nScaler Scale")
print(scaler.scale_)

print("\nPipeline Completed Successfully!")