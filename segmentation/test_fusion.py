from fusion import fuse_predictions


clinical_probability = 0.985
image_probability = 0.775


result = fuse_predictions(
    clinical_probability,
    image_probability
)

print("\n" + "=" * 50)
print("FUSION TEST")
print("=" * 50)

print(result)

print("=" * 50)