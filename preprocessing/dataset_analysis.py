import pandas as pd

# Load the dataset
df = pd.read_csv("../dataset/clinical/updated_ckd_dataset_with_stages.csv")

# Display first 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumns:")
print(df.columns.tolist())

# Display data types
print("\nData Types:")
print(df.dtypes)

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())