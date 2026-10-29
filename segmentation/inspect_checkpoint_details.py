
import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

checkpoint_names = [
    "best_unet.pth",
    "best_unet_test.pth"
]

for name in checkpoint_names:
    path = os.path.join(MODEL_DIR, name)

    print("\n" + "=" * 60)
    print("CHECKPOINT:", name)
    print("=" * 60)

    if not os.path.isfile(path):
        print("File not found:", path)
        continue

    print("File size:", round(os.path.getsize(path) / (1024 * 1024), 2), "MB")

    try:
        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        if isinstance(checkpoint, dict):
            print("Number of entries:", len(checkpoint))

            total_parameters = 0

            for key, value in checkpoint.items():
                if torch.is_tensor(value):
                    total_parameters += value.numel()

            print("Total tensor values:", total_parameters)
            print("First 10 keys:")

            for key in list(checkpoint.keys())[:10]:
                value = checkpoint[key]

                if torch.is_tensor(value):
                    print(f"  {key}: shape={tuple(value.shape)}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        else:
            print("Checkpoint object type:", type(checkpoint).__name__)

    except Exception as error:
        print("Could not inspect checkpoint:", error)

print("\nInspection complete. No checkpoint was modified.")
