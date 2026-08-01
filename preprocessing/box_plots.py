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

# Create box plots
for feature in features:
    plt.figure(figsize=(6,4))

    plt.boxplot(df[feature].dropna(), vert=True)

    plt.title(f"Box Plot of {feature}")
    plt.ylabel(feature)

    plt.grid(True)

    # Save figure
    plt.savefig(f"../reports/{feature}_boxplot.png", dpi=300)

    plt.show()

print("Box plots created successfully!")