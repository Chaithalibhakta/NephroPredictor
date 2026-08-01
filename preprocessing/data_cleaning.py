import pandas as pd

# Load dataset
df = pd.read_csv("../dataset/clinical/updated_ckd_dataset_with_stages.csv")

# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Fill missing numerical values with the median
numeric_columns = df.select_dtypes(include=["number"]).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

# Fill missing categorical values with the mode
categorical_columns = df.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("../dataset/clinical/cleaned_ckd_dataset.csv", index=False)

print("\nData cleaning completed successfully!")
print("Cleaned dataset saved as cleaned_ckd_dataset.csv")