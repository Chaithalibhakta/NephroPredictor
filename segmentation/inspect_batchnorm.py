import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

CHECKPOINTS = [
    "best_unet.pth",
    "best_unet_test.pth"
]

for checkpoint_name in CHECKPOINTS:
    path = os.path.join(MODEL_DIR, checkpoint_name)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("\n" + "=" * 75)
    print("CHECKPOINT:", checkpoint_name)
    print("=" * 75)

    for key, tensor in state.items():

        if not (
            key.endswith("running_var")
            or key.endswith("running_mean")
        ):
            continue

        values = tensor.detach().cpu().numpy()

        finite = np.isfinite(values)

        if not finite.all():
            print(
                f"\n{key}: WARNING - "
                f"{np.sum(~finite)} non-finite values"
            )
            continue

        print(
            f"{key}\n"
            f"  min    : {values.min():.6g}\n"
            f"  max    : {values.max():.6g}\n"
            f"  mean   : {values.mean():.6g}\n"
            f"  median : {np.median(values):.6g}"
        )

    nonfinite_keys = [
        key
        for key, tensor in state.items()
        if torch.is_floating_point(tensor)
        and not torch.isfinite(tensor).all()
    ]

    print("\nNon-finite floating-point tensors:", len(nonfinite_keys))

    if nonfinite_keys:
        print("Affected tensors:", nonfinite_keys)

print("\nInspection finished. No model weights were modified.")

