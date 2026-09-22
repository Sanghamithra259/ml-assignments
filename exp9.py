import os
import itertools

import matplotlib
matplotlib.use("Agg")  # write PNGs only, never try to open a display window

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

sns.set_style("whitegrid")
RANDOM_STATE = 42
PLOTS_DIR = "plot9"


# ---------------------------------------------------------------------------
# 0. Locate the dataset
# ---------------------------------------------------------------------------
def find_dataset_root(start=".", csv_name="english.csv"):
    """
    Walks down from `start` looking for english.csv.

    Returns (csv_path, dataset_root) where dataset_root is the folder that
    contains english.csv - which is also the folder the 'image' paths inside
    the CSV are relative to (they look like 'Img/img001-001.png').
    """
    for dirpath, _, filenames in os.walk(start):
        if csv_name in filenames:
            return os.path.join(dirpath, csv_name), dirpath

    raise FileNotFoundError(
        f"Could not find {csv_name} anywhere under {os.path.abspath(start)}.\n"
        "Download the English Handwritten Characters dataset from Kaggle\n"
        "(dhruvildave/english-handwritten-characters-dataset) and unzip it here."
    )


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
def load_english_chars(csv_path, dataset_root, image_col="image", label_col="label",
                       image_size=(32, 32), grayscale=True):
    """
    Loads the English Handwritten Characters dataset.

    csv_path      : path to english.csv (columns: image, label)
    dataset_root  : folder that the 'image' paths in the CSV are relative to
                    (usually the same folder that contains english.csv)
    image_size    : (width, height) to resize every image to
    grayscale     : if True, converts to single-channel before flattening

    Returns:
        X : np.ndarray, shape (n_samples, n_pixels), normalized to [0, 1]
        y : np.ndarray, string labels (e.g. '0','1',...,'A',...,'z')
        image_size : the size used (for reference / reshaping later)
    """
    df = pd.read_csv(csv_path)

    images = []
    labels = []
    missing = 0

    for _, row in df.iterrows():
        img_path = os.path.join(dataset_root, row[image_col])
        if not os.path.exists(img_path):
            missing += 1
            continue

        img = Image.open(img_path)
        img = img.convert("L") if grayscale else img.convert("RGB")
        img = img.resize(image_size)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        images.append(arr.flatten())
        labels.append(str(row[label_col]))

    if missing > 0:
        print(f"Warning: {missing} image paths from the CSV were not found and were skipped.")
        print("         (If this number is large, dataset_root is probably off by a folder level.)")

    if not images:
        raise RuntimeError(
            f"No images could be loaded. Checked paths relative to '{dataset_root}'. "
            "Expected an 'Img/' folder next to english.csv."
        )

    X = np.array(images)
    y = np.array(labels)

    print(f"Loaded {X.shape[0]} images, feature dim = {X.shape[1]}, classes = {len(set(y))}")
    return X, y, image_size


# ---------------------------------------------------------------------------
# 2. Model A: Single-Layer Perceptron (PLA), from scratch
# ---------------------------------------------------------------------------
class MulticlassPerceptron:
    """
    Single-layer perceptron with a step activation, extended to multiple
    classes via the standard multiclass mistake-driven update rule:

        scores = W @ x
        y_hat  = argmax(scores)
        if y_hat != y_true:
            W[y_true] += eta * x       # <-> the "+eta*(y - y_hat)*x" rule
            W[y_hat]  -= eta * x       # for the binary/two-class case

    This collapses to the textbook binary update  w <- w + eta(y - y_hat)x
    when there are only two classes. Each row of W is one class's weight
    vector (with a bias appended), and only the true/predicted rows change
    on a mistake - every other class's weights are untouched, exactly like
    the single-neuron rule but generalized to K outputs instead of 1.
    """

    def __init__(self, n_features, n_classes, learning_rate=0.01, epochs=50, random_state=42):
        self.n_features = n_features
        self.n_classes = n_classes
        self.lr = learning_rate
        self.epochs = epochs
        rng = np.random.RandomState(random_state)
        self.W = rng.normal(scale=0.01, size=(n_classes, n_features + 1))  # +1 for bias
        self.train_error_history_ = []
        self.val_error_history_ = []

    @staticmethod
    def _add_bias(X):
        return np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])

    def fit(self, X, y_idx, X_val=None, y_val_idx=None, verbose=True):
        """
        y_idx must already be integer class indices (0..n_classes-1).
        """
        Xb = self._add_bias(X)
        n_samples = Xb.shape[0]

        self.train_error_history_ = []
        self.val_error_history_ = []

        for epoch in range(self.epochs):
            mistakes = 0
            # shuffle each epoch (standard practice for perceptron convergence)
            perm = np.random.RandomState(epoch).permutation(n_samples)

            for i in perm:
                xi = Xb[i]
                yi = y_idx[i]
                scores = self.W @ xi
                y_hat = int(np.argmax(scores))

                if y_hat != yi:
                    self.W[yi] += self.lr * xi
                    self.W[y_hat] -= self.lr * xi
                    mistakes += 1

            train_err = mistakes / n_samples
            self.train_error_history_.append(train_err)

            if X_val is not None:
                val_pred = self.predict(X_val)
                val_err = 1.0 - accuracy_score(y_val_idx, val_pred)
                self.val_error_history_.append(val_err)

            if verbose and (epoch % 5 == 0 or epoch == self.epochs - 1):
                msg = f"Epoch {epoch+1}/{self.epochs} - train error: {train_err:.4f}"
                if X_val is not None:
                    msg += f" - val error: {self.val_error_history_[-1]:.4f}"
                print(msg)

        return self

    def predict(self, X):
        Xb = self._add_bias(X)
        scores = Xb @ self.W.T
        return np.argmax(scores, axis=1)

    def predict_scores(self, X):
        """Raw (unnormalized) class scores - used as a stand-in for 'probabilities' in ROC curves."""
        Xb = self._add_bias(X)
        return Xb @ self.W.T


def plot_pla_convergence(pla, plots_dir=PLOTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pla.train_error_history_, label="Train error", marker="o", markersize=3)
    if pla.val_error_history_:
        ax.plot(pla.val_error_history_, label="Validation error", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Misclassification rate")
    ax.set_title("PLA Training Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pla_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Model B: Multilayer Perceptron (MLP), with hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_mlp(X_train, y_train, X_val, y_val, param_grid, max_iter=100, random_state=42):
    """
    Manual grid search over MLPClassifier hyperparameters, evaluated on a
    held-out validation set (kept separate from the final test set).

    param_grid: dict of lists, e.g.
        {
            "hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "activation": ["relu", "tanh"],
            "solver": ["adam", "sgd"],
            "learning_rate_init": [0.001, 0.01],
            "batch_size": [32, 64],
        }

    Returns:
        results_df : one row per configuration tried, with val accuracy
        best_model : the fitted MLPClassifier with the best validation accuracy
        best_params : dict of its hyperparameters
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))

    results = []
    best_val_acc = -1
    best_model = None
    best_params = None

    print(f"\nTuning MLP over {len(combos)} configurations...")

    for combo in combos:
        params = dict(zip(keys, combo))

        clf = MLPClassifier(
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,
            **params
        )
        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        row = dict(params)
        row["val_accuracy"] = val_acc
        row["n_iter"] = clf.n_iter_
        row["final_loss"] = clf.loss_
        results.append(row)

        print(f"  {params} -> val_accuracy={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = clf
            best_params = params

    results_df = pd.DataFrame(results).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    print(f"\nBest MLP config: {best_params}  (val_accuracy={best_val_acc:.4f})")

    return results_df, best_model, best_params


def plot_mlp_loss_curve(mlp, plots_dir=PLOTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mlp.loss_curve_)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.set_title("MLP Training Loss Curve (best configuration)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "mlp_loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Evaluation: metrics, confusion matrix, ROC curves
# ---------------------------------------------------------------------------
def evaluate_model(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    print(f"\n{name} - Accuracy: {acc:.4f} | Precision (macro): {precision:.4f} | "
          f"Recall (macro): {recall:.4f} | F1 (macro): {f1:.4f}")
    return {"Model": name, "Accuracy": acc, "Precision": precision, "Recall": recall, "F1": f1}


def plot_confusion(y_true, y_pred, class_names, title, filename, plots_dir=PLOTS_DIR):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    n = len(class_names)
    fig_size = max(10, n * 0.3)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(cm, cmap="Blues", ax=ax, cbar=True,
                xticklabels=class_names, yticklabels=class_names,
                annot=(n <= 20), fmt="d")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(y_true_idx, y_scores, n_classes, class_names, title, filename,
                    plots_dir=PLOTS_DIR):
    """
    y_scores: array (n_samples, n_classes) of scores/probabilities
              (predict_proba for MLP, raw perceptron scores for PLA)
    Plots micro- and macro-average ROC curves (per-class curves omitted from
    the legend when there are many classes, to keep it readable).
    """
    y_true_bin = label_binarize(y_true_idx, classes=range(n_classes))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr["micro"], tpr["micro"], label=f"micro-average (AUC={roc_auc['micro']:.3f})", linewidth=2)
    ax.plot(fpr["macro"], tpr["macro"], label=f"macro-average (AUC={roc_auc['macro']:.3f})",
            linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)

    return roc_auc["micro"], roc_auc["macro"]


def plot_metric_comparison(metrics_df, plots_dir=PLOTS_DIR):
    metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(5 * len(metric_cols), 5))

    for ax, metric in zip(axes, metric_cols):
        sns.barplot(x="Model", y=metric, data=metrics_df, ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(metric)

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "ab_metric_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    plots_dir = PLOTS_DIR
    os.makedirs(plots_dir, exist_ok=True)

    # ---- CONFIG ----
    CSV_PATH, DATASET_ROOT = find_dataset_root(".")
    print(f"Using dataset at: {os.path.abspath(DATASET_ROOT)}")
    IMAGE_SIZE = (32, 32)   # resize target; flattened -> 1024 features (grayscale)

    # 1. Load & preprocess
    X, y_str, image_size = load_english_chars(CSV_PATH, DATASET_ROOT, image_size=IMAGE_SIZE)

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    class_names = le.classes_
    n_classes = len(class_names)

    # 2. Train / validation / test split (stratified, since 62 classes with ~55 images each)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(f"\nTrain: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    metrics_list = []

    # ---------------- Model A: PLA (from scratch) ----------------
    print("\n" + "=" * 60)
    print("MODEL A: Single-Layer Perceptron (PLA)")
    print("=" * 60)

    pla = MulticlassPerceptron(
        n_features=X_train.shape[1], n_classes=n_classes,
        learning_rate=0.01, epochs=50, random_state=RANDOM_STATE
    )
    pla.fit(X_train, y_train, X_val=X_val, y_val_idx=y_val)
    plot_pla_convergence(pla, plots_dir=plots_dir)

    pla_test_pred = pla.predict(X_test)
    pla_test_scores = pla.predict_scores(X_test)
    metrics_list.append(evaluate_model(y_test, pla_test_pred, name="PLA"))
    plot_confusion(y_test, pla_test_pred, class_names, "PLA - Confusion Matrix",
                   "confusion_matrix_pla.png", plots_dir)
    plot_roc_curves(y_test, pla_test_scores, n_classes, class_names,
                    "PLA - ROC Curves (micro/macro)", "roc_pla.png", plots_dir)

    # ---------------- Model B: MLP (with hyperparameter tuning) ----------------
    print("\n" + "=" * 60)
    print("MODEL B: Multilayer Perceptron (MLP) - Hyperparameter Tuning")
    print("=" * 60)

    param_grid = {
        "hidden_layer_sizes": [(128,), (128, 64), (256, 128)],
        "activation": ["relu", "tanh"],
        "solver": ["adam", "sgd"],
        "learning_rate_init": [0.001, 0.01],
        "batch_size": [32],
    }

    tuning_results, best_mlp, best_params = tune_mlp(
        X_train, y_train, X_val, y_val, param_grid, max_iter=150, random_state=RANDOM_STATE
    )
    tuning_results.to_csv(os.path.join(plots_dir, "mlp_tuning_results.csv"), index=False)
    plot_mlp_loss_curve(best_mlp, plots_dir=plots_dir)

    mlp_test_pred = best_mlp.predict(X_test)
    mlp_test_proba = best_mlp.predict_proba(X_test)
    metrics_list.append(evaluate_model(y_test, mlp_test_pred, name="MLP (tuned)"))
    plot_confusion(y_test, mlp_test_pred, class_names, "MLP - Confusion Matrix",
                   "confusion_matrix_mlp.png", plots_dir)
    plot_roc_curves(y_test, mlp_test_proba, n_classes, class_names,
                    "MLP - ROC Curves (micro/macro)", "roc_mlp.png", plots_dir)

    # ---------------- A/B Comparison ----------------
    metrics_df = pd.DataFrame(metrics_list)
    print("\nA/B Comparison (Test Set)")
    print(metrics_df)
    plot_metric_comparison(metrics_df, plots_dir=plots_dir)
    metrics_df.to_csv(os.path.join(plots_dir, "ab_comparison_summary.csv"), index=False)

    print(f"\nBest MLP hyperparameters: {best_params}")
    print(f"All plots and result tables saved to '{plots_dir}/'")


if __name__ == "__main__":
    main()