"""Data loading, downloading, dataset-selection, and camera-pose helpers."""

import os
import shutil
import subprocess
import zipfile

import numpy as np
import pandas as pd

COMPETITION = "image-matching-challenge-2025"


def load_labels(labels_path):
    """Load the ground-truth training labels CSV."""
    return pd.read_csv(labels_path)


def select_datasets(df, dataset_names):
    """Return the rows of the labels table that belong to the given datasets."""
    return df[df["dataset"].isin(dataset_names)].copy()


def get_image_names(dataset_df):
    """Return image filenames from a dataset subset."""
    return dataset_df["image"].tolist()


def summarize_datasets(df):
    """Number of images and ground-truth scenes per dataset."""
    return df.groupby("dataset").agg(
        image_count=("image", "count"),
        scene_count=("scene", "nunique"),
    ).reset_index()


def build_scene_map(df):
    """Map each image name to its ground-truth scene label."""
    return dict(zip(df["image"], df["scene"]))


def valid_pose_format(value, expected_length):
    """Check whether a semicolon-separated pose value has valid numeric entries."""
    if pd.isna(value):
        return False

    try:
        values = np.array([float(x) for x in str(value).split(";")])
        return len(values) == expected_length and np.isfinite(values).all()
    except (ValueError, TypeError):
        return False


def calculate_camera_center(rotation, translation):
    """Calculate camera center C = -R.T @ T from serialized rotation and translation."""
    rotation_values = np.array([float(x) for x in str(rotation).split(";")])
    translation_values = np.array([float(x) for x in str(translation).split(";")])

    if rotation_values.size != 9 or translation_values.size != 3:
        raise ValueError("Rotation must contain 9 values and translation must contain 3 values.")
    if not np.isfinite(rotation_values).all() or not np.isfinite(translation_values).all():
        raise ValueError("Rotation and translation values must be finite numbers.")

    rotation_matrix = rotation_values.reshape(3, 3)
    return -rotation_matrix.T @ translation_values


def download_labels(out_dir="/content"):
    """Download train_labels.csv from Kaggle (requires KAGGLE_API_TOKEN)."""
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", COMPETITION, "-f", "train_labels.csv", "-p", out_dir,
    ], check=True)


def _image_exists(folder, dataset, image_name):
    plain = os.path.join(folder, image_name)
    nested = os.path.join(folder, "train", dataset, image_name)
    return os.path.isfile(plain) or os.path.isfile(nested)


def download_dataset_images(dataset, image_names, root="/content/train"):
    """Download one dataset's images from Kaggle, skipping files already present.

    Images are stored in ``root/<dataset>``. Downloaded zip archives are
    extracted and removed. Returns the destination folder.
    """
    folder = os.path.join(root, dataset)
    os.makedirs(folder, exist_ok=True)

    for image_name in image_names:
        if _image_exists(folder, dataset, image_name):
            continue

        subprocess.run([
            "kaggle", "competitions", "download",
            "-c", COMPETITION,
            "-f", f"train/{dataset}/{image_name}",
            "-p", folder, "-q",
        ], check=True)

        zip_path = os.path.join(folder, image_name + ".zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(folder)
            os.remove(zip_path)

    return folder


def count_found_images(folder, dataset, image_names):
    """Count images found in the expected or nested folder."""
    return sum(_image_exists(folder, dataset, name) for name in image_names)


def flatten_dataset_images(folder, dataset, image_names):
    """Move Kaggle-extracted nested images to the folder expected by OpenCV."""
    nested = os.path.join(folder, "train", dataset)
    for name in image_names:
        destination = os.path.join(folder, name)
        source = os.path.join(nested, name)
        if not os.path.isfile(destination) and os.path.isfile(source):
            shutil.copy2(source, destination)
    missing = [name for name in image_names
               if not os.path.isfile(os.path.join(folder, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} images in {folder}; examples: {missing[:3]}"
        )
    return folder
