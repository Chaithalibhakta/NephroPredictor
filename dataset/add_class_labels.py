import pandas as pd

CSV_PATH = "kidney_images/kidney_dataset.csv"

df = pd.read_csv(CSV_PATH)

# Convert numeric diagnosis into meaningful class names
df["class_name"] = df["diagnosis"].map({
    0: "healthy",
    1: "pathological"
})

# Save updated CSV
df.to_csv(CSV_PATH, index=False)

print("CSV updated successfully.")

print("\nClass distribution:")
print(df["class_name"].value_counts())

print("\nDiagnosis mapping:")
print("0 = healthy")
print("1 = pathological")