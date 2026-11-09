"""Data loading and downloading helpers (Kaggle Image Matching Challenge 2025)."""
import os
import subprocess
import zipfile

import pandas as pd

COMPETITION = "image-matching-challenge-2025"


def load_labels(labels_path):
    """Load the ground-truth training labels CSV."""
    return pd.read_csv(labels_path)


def select_datasets(df, dataset_names):
    """Return the rows of the labels table that belong to the given datasets."""
    return df[df["dataset"].isin(dataset_names)].copy()


def summarize_datasets(df):
    """Number of images and ground-truth scenes per dataset."""
    return df.groupby("dataset").agg(
        image_count=("image", "count"),
        scene_count=("scene", "nunique"),
    ).reset_index()


def download_labels(out_dir="/content"):
    """Download train_labels.csv from Kaggle (requires KAGGLE_API_TOKEN)."""
    subprocess.run([
        "kaggle", "competitions", "download",
        "-c", COMPETITION, "-f", "train_labels.csv", "-p", out_dir,
    ], check=True)


def _image_exists(folder, dataset, image_name):
    plain = os.path.join(folder, image_name)
    nested = os.path.join(folder, "train", dataset, image_name)
    return os.path.isfile(plain) or os.path.isfile(nested)


def download_dataset_images(dataset, image_names, root="/content/train"):
    """Download the images of one training dataset from Kaggle.

    Images are stored in ``root/<dataset>``. Files that already exist are
    skipped, and downloaded zip archives are extracted and removed.
    Returns the destination folder.
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
    """Count how many images exist in the expected or the nested folder."""
    return sum(_image_exists(folder, dataset, n) for n in image_names)


def build_scene_map(df):
    """Map each image name to its ground-truth scene label."""
    return dict(zip(df["image"], df["scene"]))
