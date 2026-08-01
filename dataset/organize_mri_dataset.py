import os
import shutil

# -----------------------------
# CHANGE THESE PATHS IF NEEDED
# -----------------------------
images_path = "MRI/images"
masks_path = "MRI/masks"


def flatten_folder(main_folder):
    print(f"\nProcessing: {main_folder}")

    for item in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, item)

        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".nii") or file.endswith(".nii.gz"):

                    source = os.path.join(folder_path, file)
                    destination = os.path.join(main_folder, file)

                    if not os.path.exists(destination):
                        shutil.move(source, destination)
                        print(f"Moved: {file}")
                    else:
                        print(f"Already exists: {file}")

            # Remove empty folder
            try:
                os.rmdir(folder_path)
                print(f"Removed empty folder: {item}")
            except:
                pass


flatten_folder(images_path)
flatten_folder(masks_path)

print("\n==============================")
print("MRI Dataset Organized Successfully!")
print("==============================")