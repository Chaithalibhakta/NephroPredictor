import pandas as pd

# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("../dataset/clinical/updated_ckd_dataset_with_stages.csv")

print("=" * 50)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 50)

print("\nShape Before Cleaning:", df.shape)

# ==========================================
# Missing Values
# ==========================================
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# ==========================================
# Remove Duplicates
# ==========================================
duplicates = df.duplicated().sum()
print("\nDuplicate Rows:", duplicates)

df = df.drop_duplicates()

# ==========================================
# Fill Missing Numerical Values
# ==========================================
numeric_columns = df.select_dtypes(include=["number"]).columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

# ==========================================
# Fill Missing Categorical Values
# ==========================================
categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================================
# Verify Cleaning
# ==========================================
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nShape After Cleaning:", df.shape)

# ==========================================
# Save Clean Dataset
# ==========================================
output_path = "../dataset/clinical/cleaned_ckd_dataset.csv"

df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Location:", output_path)

print("\nFirst 5 Rows:")
print(df.head())

print("\nCleaning Completed Successfully!")