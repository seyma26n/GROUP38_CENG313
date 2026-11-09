"""Baseline models: feature extraction, matching and graph clustering."""
import os
from itertools import combinations

import cv2
import networkx as nx
import pandas as pd


def _extract_features(detector, image_names, image_folder):
    features = {}
    for name in image_names:
        image_path = os.path.join(image_folder, name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        keypoints, descriptors = detector.detectAndCompute(image, None)
        features[name] = {"keypoints": keypoints, "descriptors": descriptors}
    return features


def extract_sift_features(image_names, image_folder, nfeatures=1000):
    """Extract SIFT keypoints and descriptors from grayscale images."""
    return _extract_features(cv2.SIFT_create(nfeatures=nfeatures),
                             image_names, image_folder)


def extract_orb_features(image_names, image_folder, nfeatures=1000):
    """Extract ORB keypoints and binary descriptors from grayscale images."""
    return _extract_features(cv2.ORB_create(nfeatures=nfeatures),
                             image_names, image_folder)


def _match_all_pairs(matcher, image_names, features, ratio):
    rows = []
    for name1, name2 in combinations(image_names, 2):
        des1 = features[name1]["descriptors"]
        des2 = features[name2]["descriptors"]
        good = 0
        if des1 is not None and des2 is not None and len(des2) >= 2:
            for pair in matcher.knnMatch(des1, des2, k=2):
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                    good += 1  # Lowe's ratio test
        rows.append({"image1": name1, "image2": name2, "good_matches": good})
    return pd.DataFrame(rows)


def match_sift_features(image_names, features, ratio=0.75):
    """Match SIFT descriptors of all image pairs with FLANN + ratio test."""
    cv2.setRNGSeed(42)  # more reproducible matching
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    return _match_all_pairs(flann, image_names, features, ratio)


def match_orb_features(image_names, features, ratio=0.75):
    """Match ORB descriptors of all image pairs with BFMatcher (Hamming)."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    return _match_all_pairs(bf, image_names, features, ratio)


def cluster_images(image_names, pairs_df, min_matches=20, resolution=1.0, seed=42):
    """Build a weighted match graph and run weighted Louvain clustering.

    Every image is a node; an edge links two images with at least
    ``min_matches`` good matches, weighted by the match count.
    Returns ``(graph, clusters)``.
    """
    G = nx.Graph()
    G.add_nodes_from(image_names)
    for _, row in pairs_df.iterrows():
        if row["good_matches"] >= min_matches:
            G.add_edge(row["image1"], row["image2"],
                       weight=float(row["good_matches"]))
    clusters = nx.community.louvain_communities(
        G, weight="weight", resolution=resolution, seed=seed)
    return G, clusters
