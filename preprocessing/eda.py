import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../dataset/clinical/cleaned_ckd_dataset.csv")

# Dataset information
print(df.info())

# Plot CKD Stage Distribution
plt.figure(figsize=(8,5))
df["ckd_stage"].value_counts().plot(kind="bar")
plt.title("CKD Stage Distribution")
plt.xlabel("CKD Stage")
plt.ylabel("Number of Patients")
plt.tight_layout()
plt.savefig("../reports/ckd_stage_distribution.png")
plt.show()