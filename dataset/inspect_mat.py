import scipy.io as sio
import os

mat_path = "kidney_images/masks/1.mat"

print("Reading:", mat_path)

data = sio.loadmat(mat_path)

print("\nVariables inside 1.mat:")
print("=" * 40)

for key, value in data.items():

    if not key.startswith("__"):

        print("\nVariable:", key)
        print("Type:", type(value))
        print("Shape:", getattr(value, "shape", "N/A"))
        print("Data type:", getattr(value, "dtype", "N/A"))

        if hasattr(value, "min"):
            print("Minimum:", value.min())

        if hasattr(value, "max"):
            print("Maximum:", value.max())