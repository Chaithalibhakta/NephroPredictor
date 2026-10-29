
import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

path_a = os.path.join(MODEL_DIR, "best_unet.pth")
path_b = os.path.join(MODEL_DIR, "best_unet_test.pth")

model_a = torch.load(
    path_a,
    map_location="cpu",
    weights_only=True
)

model_b = torch.load(
    path_b,
    map_location="cpu",
    weights_only=True
)

print("=" * 65)
print("U-NET CHECKPOINT WEIGHT COMPARISON")
print("=" * 65)

if model_a.keys() != model_b.keys():
    print("ERROR: Checkpoint keys do not match.")
    raise SystemExit(1)

total_values = 0
different_values = 0
sum_squared_diff = 0.0
sum_squared_a = 0.0

layer_results = []

for key in model_a:
    a = model_a[key].float()
    b = model_b[key].float()

    difference = a - b

    count = a.numel()
    changed = int(torch.count_nonzero(difference).item())

    total_values += count
    different_values += changed

    sum_squared_diff += torch.sum(difference ** 2).item()
    sum_squared_a += torch.sum(a ** 2).item()

    layer_results.append((
        key,
        count,
        changed,
        difference.abs().max().item()
    ))

rmse = np.sqrt(sum_squared_diff / total_values)
relative_difference = (
    np.sqrt(sum_squared_diff / sum_squared_a)
    if sum_squared_a > 0 else 0
)

print("\nTotal tensor values:", total_values)
print("Values that differ:", different_values)
print(
    "Percentage of values that differ:",
    round(100 * different_values / total_values, 4), "%"
)
print("Overall weight RMSE:", round(rmse, 8))
print(
    "Relative weight difference:",
    round(100 * relative_difference, 4), "%"
)

print("\nLayers with the largest maximum absolute differences:")

layer_results.sort(key=lambda x: x[3], reverse=True)

for key, count, changed, max_diff in layer_results[:15]:
    print(
        f"{key}: values={count}, "
        f"different={changed}, "
        f"max_abs_diff={max_diff:.8f}"
    )

print("\nComparison complete. Neither checkpoint was modified.")

