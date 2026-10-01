import torch
from unet_model import UNet


print("=" * 50)
print("TESTING U-NET")
print("=" * 50)

# Create model
model = UNet()

print("\nU-Net created successfully.")

# Create fake input
x = torch.randn(
    1,          # batch
    3,          # RGB channels
    256,        # height
    256         # width
)

# Forward pass
with torch.no_grad():

    output = model(x)

print("\nInput shape :")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

# Verify expected output
expected = (
    1,
    1,
    256,
    256
)

if tuple(output.shape) == expected:

    print("\n✅ U-NET TEST PASSED")

else:

    print("\n❌ U-NET TEST FAILED")