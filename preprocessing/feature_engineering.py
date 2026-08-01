import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load cleaned dataset
df = pd.read_csv("../dataset/clinical/cleaned_ckd_dataset.csv")

# ----------------------------
# Encode categorical columns
# ----------------------------
label_encoder = LabelEncoder()

categorical_columns = df.select_dtypes(include=["object"]).columns

for col in categorical_columns:
    df[col] = label_encoder.fit_transform(df[col])

print("Categorical features encoded successfully!")

# ----------------------------
# Scale numerical features
# ----------------------------
scaler = StandardScaler()

# Exclude target columns
exclude_columns = ["ckd_pred", "ckd_stage"]

numeric_columns = [col for col in df.columns if col not in exclude_columns]

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("Numerical features scaled successfully!")

# ----------------------------
# Save processed dataset
# ----------------------------
df.to_csv("../dataset/clinical/processed_ckd_dataset.csv", index=False)

print("Processed dataset saved successfully!")