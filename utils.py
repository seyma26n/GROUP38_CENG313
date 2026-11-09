"""Evaluation metrics and plotting helpers."""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, precision_recall_fscore_support


def evaluate_clustering(clusters, image_names, scene_map):
    """Compare predicted clusters with ground-truth scenes.

    Returns ARI and pairwise precision, recall and F1.
    """
    predicted_map = {name: cid for cid, cluster in enumerate(clusters)
                     for name in cluster}

    true_labels = [scene_map[n] for n in image_names]
    pred_labels = [predicted_map[n] for n in image_names]
    ari = adjusted_rand_score(true_labels, pred_labels)

    y_true, y_pred = [], []
    for a, b in combinations(image_names, 2):
        y_true.append(int(scene_map[a] == scene_map[b]))
        y_pred.append(int(predicted_map[a] == predicted_map[b]))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)

    return {"ARI": round(ari, 4), "Precision": round(precision, 4),
            "Recall": round(recall, 4), "Pairwise F1": round(f1, 4)}


def cluster_composition(clusters, scene_map):
    """Cross-tabulation of predicted clusters versus true scenes."""
    rows = [{"Predicted Cluster": cid, "True Scene": scene_map[name]}
            for cid, cluster in enumerate(clusters) for name in cluster]
    df = pd.DataFrame(rows)
    return pd.crosstab(df["Predicted Cluster"], df["True Scene"])


def plot_method_comparison(comparison_df, datasets=("Stairs", "Haiper"),
                           metrics=("ARI", "Pairwise F1")):
    """Bar charts comparing SIFT and ORB for each metric."""
    for metric in metrics:
        x = np.arange(len(datasets))
        width = 0.35
        values = {
            method: [comparison_df[(comparison_df["Dataset"] == d) &
                                   (comparison_df["Method"] == method)][metric].values[0]
                     for d in datasets]
            for method in ("SIFT", "ORB")
        }
        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, values["SIFT"], width, label="SIFT")
        plt.bar(x + width / 2, values["ORB"], width, label="ORB")
        plt.xticks(x, datasets)
        plt.ylabel(metric)
        plt.title(f"SIFT vs ORB - {metric} Comparison")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()
