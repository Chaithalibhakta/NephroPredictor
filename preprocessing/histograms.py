import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("../dataset/clinical/cleaned_ckd_dataset.csv")

# Important numerical features
features = [
    "serum_creatinine",
    "gfr",
    "bun",
    "serum_calcium",
    "urine_ph",
    "blood_pressure"
]

# Create histogram for each feature
for feature in features:
    plt.figure(figsize=(6,4))
    plt.hist(df[feature], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.grid(True)

    # Save graph
    plt.savefig(f"../reports/{feature}_histogram.png")
    plt.show()

print("Histograms created successfully!")