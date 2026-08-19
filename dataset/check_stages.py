import pandas as pd

df = pd.read_csv("../dataset/clinical/processed_ckd_dataset.csv")

print(df["ckd_stage"].value_counts())