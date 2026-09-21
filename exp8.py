"""
ICS1512 - Machine Learning Algorithms Laboratory
Experiment 8: Clustering Human Activity Recognition Data using
K-Means, DBSCAN, and Hierarchical Clustering
"""

import os
import math

import matplotlib
matplotlib.use("Agg")  # non-interactive backend: write PNGs, never open a display window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix,
)

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Your existing EDA function (kept as-is)
# ---------------------------------------------------------------------------
def perform_eda(file_path, target_col=None, plots_dir="plots", show=False,
                 remove_outliers=True, max_plot_cols=None, max_corr_cols=None):
    """
    remove_outliers : if False, skips the per-column IQR filtering step entirely.
                       Needed for wide datasets (e.g. HAR's 561 features) where
                       ANDing IQR masks across every column can drop nearly all rows.
    max_plot_cols   : if set and there are more numeric columns than this, only the
                       first max_plot_cols are plotted in the distribution grid
                       (prevents a huge/failing figure on wide datasets).
    max_corr_cols   : if set and there are more numeric columns than this, the
                       correlation heatmap is limited to the first max_corr_cols
                       columns (prevents a huge/failing NxN heatmap).
    Both new params default to the original behavior (remove_outliers=True,
    no plot column cap) so existing calls are unaffected.
    """

    os.makedirs(plots_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Preview
    print("\nFirst 5 Rows")
    print(df.head())

    # Dataset information
    print("\nDataset Information")
    df.info()

    # Shape
    print("\nDataset Shape:", df.shape)

    # Column names
    columns = df.columns.tolist()
    print("\nColumn Names")
    print(columns)

    # Replace null-like placeholder strings with actual NaN
    null_values = [
        "No Info", "Noinfo", "No Data", "None",
        "NULL", "Null", "null",
        "N/A", "NA",
        "NaN", "nan",
        "Unknown", "unknown",
        "?", "", " ", "-", "--"
    ]
    df.replace(null_values, np.nan, inplace=True)

    # Remove leading/trailing spaces (only on true string cells)
    object_cols = df.select_dtypes(include="object").columns
    for col in object_cols:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Missing values
    print("\nMissing Values Before Cleaning")
    print(df.isnull().sum())

    df = df.dropna()

    print("\nMissing Values After Cleaning")
    print(df.isnull().sum())

    # Duplicate values
    duplicates = df.duplicated().sum()
    print("\nDuplicate Rows:", duplicates)

    df = df.drop_duplicates()
    print("Shape After Removing Duplicates:", df.shape)

    # Data types
    print("\nData Types")
    print(df.dtypes)

    # Unique values
    print("\nUnique Values Per Column")
    print(df.nunique())

    # Descriptive statistics
    print("\nNumerical Statistics")
    print(df.describe())

    if len(df.select_dtypes(include="object").columns) > 0:
        print("\nCategorical Statistics")
        print(df.describe(include="object"))

    # Separate columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if target_col is not None and target_col in numeric_cols:
        numeric_cols.remove(target_col)
        categorical_cols.append(target_col)

    print("\nNumeric Columns (features only)")
    print(numeric_cols)

    print("\nCategorical Columns (incl. target if set)")
    print(categorical_cols)

    if remove_outliers:
        print("\nRemoving Outliers (IQR Method)")

        original_shape = df.shape
        keep_mask = pd.Series(True, index=df.index)

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            col_mask = (df[col] >= lower) & (df[col] <= upper)
            outliers = (~col_mask).sum()
            print(f"{col}: {outliers} outliers flagged")

            keep_mask &= col_mask

        df = df[keep_mask]

        print("\nShape Before Outlier Removal:", original_shape)
        print("Shape After Outlier Removal:", df.shape)

        if df.empty:
            print("\n All rows were removed by outlier filtering. "
                  "Consider relaxing the IQR multiplier or excluding more columns.")
            return df
    else:
        print("\nSkipping IQR outlier removal (remove_outliers=False) — "
              "too many numeric columns for per-column AND-filtering to be meaningful.")

    # Numerical graphs
    plot_numeric_cols = numeric_cols
    if max_plot_cols is not None and len(numeric_cols) > max_plot_cols:
        print(f"\n{len(numeric_cols)} numeric columns is too many to plot individually; "
              f"showing distributions for the first {max_plot_cols} only.")
        plot_numeric_cols = numeric_cols[:max_plot_cols]

    if len(plot_numeric_cols) > 0:
        cols = 2
        rows = math.ceil(len(plot_numeric_cols) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for i, col in enumerate(plot_numeric_cols):
            sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"{col} Distribution")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Numeric Feature Distributions", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, "eda_numeric_distributions.png"),
            dpi=200,
            bbox_inches="tight"
        )
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Categorical graphs
    if len(categorical_cols) > 0:
        cols = 2
        rows = math.ceil(len(categorical_cols) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for i, col in enumerate(categorical_cols):
            sns.countplot(
                x=col,
                data=df,
                order=df[col].value_counts().index,
                ax=axes[i]
            )
            axes[i].set_title(f"{col} Distribution")
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle("Categorical Feature Distributions", fontsize=16)
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, "eda_categorical_distributions.png"),
            dpi=200,
            bbox_inches="tight"
        )
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Correlation matrix
    corr_cols = numeric_cols
    if max_corr_cols is not None and len(numeric_cols) > max_corr_cols:
        print(f"\n{len(numeric_cols)} numeric columns is too many for a legible "
              f"correlation heatmap; showing the first {max_corr_cols} only.")
        corr_cols = numeric_cols[:max_corr_cols]

    if len(corr_cols) > 1:
        n = len(corr_cols)
        fig_w = max(12, n * 0.55)
        fig_h = max(10, n * 0.5)
        annot_font_size = 6 if n > 20 else (8 if n > 12 else 10)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            df[corr_cols].corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            annot_kws={"size": annot_font_size},
            ax=ax
        )
        ax.set_title("Correlation Matrix")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(
            os.path.join(plots_dir, "eda_correlation_matrix.png"),
            dpi=200,
            bbox_inches="tight"
        )
        if show:
            plt.show()
        else:
            plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# 0. UCI HAR raw dataset loader (X_train.txt / y_train.txt / features.txt layout)
# ---------------------------------------------------------------------------
def load_uci_har(dataset_root, combine_train_test=True, out_csv="har_combined.csv"):
    """
    Loads the official UCI HAR Dataset folder structure:

        dataset_root/
            features.txt
            activity_labels.txt
            train/X_train.txt, y_train.txt, subject_train.txt
            test/X_test.txt, y_test.txt, subject_test.txt

    Builds one combined DataFrame with named feature columns, an "Activity"
    column (string labels, e.g. WALKING), and a "subject" column. Also
    writes it out to out_csv so perform_eda() (which expects a CSV path)
    can be pointed at it directly.

    Returns the path to the written CSV.
    """
    features = pd.read_csv(
        os.path.join(dataset_root, "features.txt"),
        sep=r"\s+", header=None, names=["idx", "feature"]
    )
    # Feature names in this dataset aren't unique (e.g. many "mean()" cols) -
    # dedupe by appending a running counter so pandas doesn't collide.
    feat_names = features["feature"].tolist()
    seen = {}
    unique_feat_names = []
    for name in feat_names:
        seen[name] = seen.get(name, 0) + 1
        unique_feat_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    activity_labels = pd.read_csv(
        os.path.join(dataset_root, "activity_labels.txt"),
        sep=r"\s+", header=None, names=["id", "activity"]
    )
    id_to_activity = dict(zip(activity_labels["id"], activity_labels["activity"]))

    def load_split(split):
        X = pd.read_csv(
            os.path.join(dataset_root, split, f"X_{split}.txt"),
            sep=r"\s+", header=None, names=unique_feat_names
        )
        y = pd.read_csv(
            os.path.join(dataset_root, split, f"y_{split}.txt"),
            sep=r"\s+", header=None, names=["activity_id"]
        )
        subj = pd.read_csv(
            os.path.join(dataset_root, split, f"subject_{split}.txt"),
            sep=r"\s+", header=None, names=["subject"]
        )
        X["Activity"] = y["activity_id"].map(id_to_activity)
        X["subject"] = subj["subject"]
        return X

    if combine_train_test:
        df = pd.concat([load_split("train"), load_split("test")], ignore_index=True)
    else:
        df = load_split("train")

    df.to_csv(out_csv, index=False)
    print(f"Combined UCI HAR data written to: {out_csv}  (shape={df.shape})")
    return out_csv


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------
def load_and_preprocess(features_path, labels_path=None, label_col="Activity"):
    """
    Loads the HAR dataset. Two common layouts are supported:
      (a) a single CSV that already has a label column (label_col)
      (b) separate feature file + label file (e.g. X_train.txt / y_train.txt style,
          already converted to CSV)

    Returns:
        X_scaled : np.ndarray, standardized feature matrix
        y_true   : np.ndarray or None, encoded true activity labels (for external metrics)
        feature_names : list of column names used
        le : fitted LabelEncoder (or None)
    """
    df = pd.read_csv(features_path)

    y_true = None
    le = None

    if labels_path is not None:
        y_raw = pd.read_csv(labels_path, header=None).iloc[:, 0].values
        le = LabelEncoder()
        y_true = le.fit_transform(y_raw)
        feature_df = df
    elif label_col in df.columns:
        y_raw = df[label_col].values
        le = LabelEncoder()
        y_true = le.fit_transform(y_raw)
        feature_df = df.drop(columns=[label_col])
        # drop subject/id-like columns if present
        for id_col in ["subject", "Subject", "ID", "id"]:
            if id_col in feature_df.columns:
                feature_df = feature_df.drop(columns=[id_col])
    else:
        feature_df = df

    # Keep numeric columns only, drop rows with NaN
    feature_df = feature_df.select_dtypes(include=[np.number]).dropna()
    feature_names = feature_df.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)

    return X_scaled, y_true, feature_names, le


# ---------------------------------------------------------------------------
# 2. K-Means with Elbow Method + Silhouette
# ---------------------------------------------------------------------------
def run_kmeans_elbow(X, k_range=range(2, 9), plots_dir="plots", random_state=42):
    wcss = []
    sil_scores = []

    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X)
        wcss.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    results_df = pd.DataFrame({
        "k": list(k_range),
        "WCSS": wcss,
        "Silhouette": sil_scores
    })
    print("\nK-Means Elbow / Silhouette Results")
    print(results_df)

    # Elbow curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_range), wcss, marker="o")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("WCSS (Inertia)")
    ax.set_title("Elbow Method")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "kmeans_elbow_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Silhouette curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(list(k_range), sil_scores, marker="o", color="darkorange")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score vs k")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "kmeans_silhouette_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    best_k = results_df.loc[results_df["Silhouette"].idxmax(), "k"]
    print(f"\nSuggested best k (by silhouette): {best_k}")

    return results_df, int(best_k)


def fit_kmeans(X, k, random_state=42):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km


# ---------------------------------------------------------------------------
# 3. DBSCAN
# ---------------------------------------------------------------------------
def run_dbscan(X, eps=0.5, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    print(f"\nDBSCAN (eps={eps}, min_samples={min_samples})")
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({100 * n_noise / len(labels):.2f}%)")

    return labels, db


def tune_dbscan_eps(X, k=5, plots_dir="plots"):
    """
    k-distance plot to help pick eps: sorts the distance to the k-th
    nearest neighbor for every point.
    """
    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_distances)
    ax.set_xlabel("Points sorted by distance")
    ax.set_ylabel(f"{k}-th Nearest Neighbor Distance")
    ax.set_title("k-Distance Plot (for choosing DBSCAN eps)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "dbscan_kdistance_plot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return k_distances


# ---------------------------------------------------------------------------
# 4. Hierarchical Agglomerative Clustering
# ---------------------------------------------------------------------------
def run_hierarchical(X, n_clusters, linkage_method="ward"):
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = hac.fit_predict(X)
    return labels, hac


def plot_dendrogram(X, plots_dir="plots", method="ward", sample_size=500, random_state=42):
    """
    Dendrograms are expensive on large HAR datasets, so subsample if needed.
    """
    if X.shape[0] > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    Z = linkage(X_sample, method=method)

    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90)
    ax.set_title(f"Dendrogram ({method} linkage, n={X_sample.shape[0]} sample)")
    ax.set_xlabel("Cluster size / sample index")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"dendrogram_{method}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Dimensionality reduction for visualization
# ---------------------------------------------------------------------------
def reduce_dimensions(X, method="pca", n_components=2, random_state=42):
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        X_reduced = reducer.fit_transform(X)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state,
                        init="pca", perplexity=30)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    return X_reduced, reducer


def plot_clusters_2d(X_2d, labels, title, filename, plots_dir="plots"):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best", fontsize=8)
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate_clustering(X, labels, y_true=None, name="Model"):
    result = {"Model": name}

    # Internal metrics need at least 2 clusters and no single giant "noise" cluster
    valid_mask = labels != -1
    unique_labels = set(labels[valid_mask]) if -1 in labels else set(labels)

    if len(unique_labels) > 1 and valid_mask.sum() > 1:
        X_valid = X[valid_mask] if -1 in labels else X
        labels_valid = labels[valid_mask] if -1 in labels else labels

        result["Silhouette"] = silhouette_score(X_valid, labels_valid)
        result["Davies-Bouldin"] = davies_bouldin_score(X_valid, labels_valid)
        result["Calinski-Harabasz"] = calinski_harabasz_score(X_valid, labels_valid)
    else:
        result["Silhouette"] = np.nan
        result["Davies-Bouldin"] = np.nan
        result["Calinski-Harabasz"] = np.nan

    if y_true is not None:
        result["ARI"] = adjusted_rand_score(y_true, labels)
        result["NMI"] = normalized_mutual_info_score(y_true, labels)
    else:
        result["ARI"] = np.nan
        result["NMI"] = np.nan

    return result


def plot_metric_comparison(metrics_df, plots_dir="plots"):
    metric_cols = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "ARI", "NMI"]
    available = [c for c in metric_cols if c in metrics_df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        sns.barplot(x="Model", y=metric, data=metrics_df, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "metric_comparison_barplot.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, cluster_labels, class_names, plots_dir="plots",
                           filename="confusion_matrix.png", title="Cluster vs Activity"):
    """
    Maps each cluster to its majority true class, then plots a confusion matrix.
    Only meaningful when y_true is available.
    """
    cm = confusion_matrix(y_true, cluster_labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"C{c}" for c in sorted(set(cluster_labels))],
                yticklabels=class_names)
    ax.set_xlabel("Predicted Cluster")
    ax.set_ylabel("True Activity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # ---- CONFIG: point this at your local UCI HAR Dataset folder ----
    # e.g. "/home/sanghami/ML-EXP1/UCI HAR Dataset/UCI HAR Dataset"
    UCI_HAR_ROOT = "UCI HAR Dataset/UCI HAR Dataset"

    FEATURES_PATH = load_uci_har(UCI_HAR_ROOT, combine_train_test=True,
                                  out_csv="har_combined.csv")
    LABELS_PATH = None                # labels already merged into FEATURES_PATH as "Activity"
    LABEL_COL = "Activity"

    # 1. EDA (using your existing function)
    # HAR has 561 numeric feature columns, so:
    #  - remove_outliers=False: skip per-column IQR filtering (it ANDs across all
    #    561 columns and will drop almost every row otherwise)
    #  - max_plot_cols / max_corr_cols: cap the distribution grid and correlation
    #    heatmap so matplotlib isn't asked to render a 1000+ inch figure
    df = perform_eda(
        FEATURES_PATH, target_col=LABEL_COL, plots_dir=plots_dir,
        remove_outliers=False, max_plot_cols=30, max_corr_cols=40
    )

    # 2. Preprocess for clustering (scale numeric features, encode labels)
    X, y_true, feature_names, le = load_and_preprocess(
        FEATURES_PATH, labels_path=LABELS_PATH, label_col=LABEL_COL
    )
    print(f"\nFeature matrix shape after preprocessing: {X.shape}")

    # 3. PCA / t-SNE for visualization (also useful as pre-clustering EDA)
    X_pca, _ = reduce_dimensions(X, method="pca")
    # t-SNE is slow on the full HAR dataset; subsample if it's large
    tsne_input = X if X.shape[0] <= 3000 else X[np.random.RandomState(42).choice(X.shape[0], 3000, replace=False)]
    X_tsne, _ = reduce_dimensions(tsne_input, method="tsne")

    metrics_list = []

    # ---------------- Model A: K-Means ----------------
    elbow_df, best_k = run_kmeans_elbow(X, k_range=range(2, 9), plots_dir=plots_dir)
    kmeans_labels, kmeans_model = fit_kmeans(X, k=best_k)
    plot_clusters_2d(X_pca, kmeans_labels, f"K-Means (k={best_k}) - PCA", "kmeans_pca.png", plots_dir)
    metrics_list.append(evaluate_clustering(X, kmeans_labels, y_true, name="K-Means"))

    # ---------------- Model B: DBSCAN ----------------
    tune_dbscan_eps(X, k=5, plots_dir=plots_dir)          # inspect this plot, then set eps below
    dbscan_labels, dbscan_model = run_dbscan(X, eps=3.0, min_samples=10)  # TUNE eps/min_samples
    plot_clusters_2d(X_pca, dbscan_labels, "DBSCAN - PCA", "dbscan_pca.png", plots_dir)
    metrics_list.append(evaluate_clustering(X, dbscan_labels, y_true, name="DBSCAN"))

    # ---------------- Model C: Hierarchical (Ward) ----------------
    plot_dendrogram(X, plots_dir=plots_dir, method="ward")
    n_clusters_hac = best_k if y_true is None else len(np.unique(y_true))
    hac_labels, hac_model = run_hierarchical(X, n_clusters=n_clusters_hac, linkage_method="ward")
    plot_clusters_2d(X_pca, hac_labels, f"Hierarchical Ward (k={n_clusters_hac}) - PCA", "hac_pca.png", plots_dir)
    metrics_list.append(evaluate_clustering(X, hac_labels, y_true, name="Hierarchical (Ward)"))

    # ---------------- Compare all models ----------------
    metrics_df = pd.DataFrame(metrics_list)
    print("\nClustering Evaluation Metrics")
    print(metrics_df)
    plot_metric_comparison(metrics_df, plots_dir=plots_dir)

    if y_true is not None and le is not None:
        plot_confusion_matrix(y_true, kmeans_labels, le.classes_, plots_dir=plots_dir,
                               filename="confusion_matrix_kmeans.png", title="K-Means vs Activity")

    metrics_df.to_csv(os.path.join(plots_dir, "clustering_metrics_summary.csv"), index=False)
    print(f"\nAll plots and metrics saved to '{plots_dir}/'")


if __name__ == "__main__":
    main()