import os
import tarfile
from pathlib import Path

# Paths
root_dir = Path("/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2")
output_dir = Path("/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2_tar")
output_dir.mkdir(parents=True, exist_ok=True)


def make_tar_from_folder(folder: Path, tar_path: Path):
    """Create a .tar file from all files in a folder, preserving hierarchy relative to root_dir."""
    with tarfile.open(tar_path, "w") as tar:
        for path in folder.rglob("*"):
            if path.is_file():
                arcname = path.relative_to(root_dir)  # keep structure
                tar.add(path, arcname=str(arcname))
    print(f"Created {tar_path}")


# Loop over first-level subdirectories
for sub in root_dir.iterdir():
    if sub.is_dir():
        if sub.name == "processed":
            # Go one level deeper (train/val/test)
            for split in sub.iterdir():
                if split.is_dir():
                    tar_path = output_dir / f"{split.name}.tar"
                    make_tar_from_folder(split, tar_path)
        else:
            tar_path = output_dir / f"{sub.name}.tar"
            make_tar_from_folder(sub, tar_path)
    elif sub.is_file():
        # If there are stray files directly under root_dir
        tar_path = output_dir / f"{sub.name}.tar"
        with tarfile.open(tar_path, "w") as tar:
            arcname = sub.relative_to(root_dir)
            tar.add(sub, arcname=str(arcname))
        print(f"Created {tar_path}")

