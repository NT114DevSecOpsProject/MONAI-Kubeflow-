from monai.apps import download_and_extract
import os

print("Downloading MedNIST dataset...")
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir, "data")

download_and_extract(
    url="https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/MedNIST.tar.gz",
    output_dir=data_dir,
)

print(f"Data downloaded to: {data_dir}")
print("Done!")