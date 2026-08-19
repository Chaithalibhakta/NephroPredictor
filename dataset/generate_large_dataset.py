import pandas as pd
import numpy as np

np.random.seed(42)

N = 500000

data = pd.DataFrame()

# Numerical Features
data["serum_creatinine"] = np.random.normal(1.5,1.2,N).clip(0.3,12)
data["gfr"] = np.random.normal(55,35,N).clip(5,130)
data["bun"] = np.random.normal(40,40,N).clip(5,250)
data["serum_calcium"] = np.random.normal(8.5,1.4,N).clip(5,12)
data["oxalate_levels"] = np.random.normal(2.5,1,N).clip(0.5,8)
data["urine_ph"] = np.random.normal(6.4,1,N).clip(4,9)
data["blood_pressure"] = np.random.normal(120,25,N).clip(70,220)

# Binary Features
data["ana"] = np.random.randint(0,2,N)
data["c3_c4"] = np.random.randint(0,2,N)
data["hematuria"] = np.random.randint(0,2,N)
data["smoking"] = np.random.randint(0,2,N)
data["alcohol"] = np.random.randint(0,2,N)
data["painkiller_usage"] = np.random.randint(0,2,N)
data["family_history"] = np.random.randint(0,2,N)

# Lifestyle
data["physical_activity"] = np.random.randint(0,3,N)
data["diet"] = np.random.randint(0,3,N)
data["water_intake"] = np.random.uniform(1,5,N)
data["weight_changes"] = np.random.randint(0,3,N)
data["stress_level"] = np.random.randint(1,6,N)
data["months"] = np.random.randint(1,25,N)

# -----------------------
# CKD Prediction Logic
# -----------------------

score = (
    (data["serum_creatinine"]>2).astype(int)
    +(data["gfr"]<60).astype(int)
    +(data["bun"]>40).astype(int)
    +(data["blood_pressure"]>140).astype(int)
    +(data["hematuria"]).astype(int)
    +(data["family_history"]).astype(int)
)

data["ckd_pred"] = np.where(score>=3,"CKD","No CKD")

# -----------------------
# Stage
# -----------------------

stage=[]

for g in data["gfr"]:

    if g>=90:
        stage.append(1)
    elif g>=60:
        stage.append(2)
    elif g>=30:
        stage.append(3)
    elif g>=15:
        stage.append(4)
    else:
        stage.append(5)

data["ckd_stage"]=stage

data.to_csv(
    "clinical/updated_ckd_dataset_with_stages.csv",
    index=False
)

print("="*60)
print("500000 Dataset Generated Successfully")
print(data.shape)
print("="*60)