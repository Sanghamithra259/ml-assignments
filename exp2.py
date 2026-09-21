from eda_function import perform_eda
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Output directory for plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def savefig(fig, filename):
    """Save a figure at report-quality resolution."""
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=300, bbox_inches="tight")



# Data
df = perform_eda("spambase_csv.csv")

X_raw = df.drop("class", axis=1)
y = df["class"]

if y.dtype == object:
    y = y.map({"ham": 0, "spam": 1})

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

X_all_raw = X_raw
y_all = y


def evaluate(name, model, X_test, y_test, ax_cm, ax_roc, ax_pr):
    """Fit-independent evaluation: draws CM/ROC/PR into given axes
    (part of the single combined dashboard figure) and returns metrics."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    for key in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        print(f"{key:<10}: {metrics[key]:.4f}")

    ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax_cm, colorbar=False)
    ax_cm.set_title(f"{name}\nConfusion Matrix", fontsize=9)

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc)
    ax_roc.set_title(f"{name}\nROC Curve", fontsize=9)
    ax_roc.legend(fontsize=7)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax_pr)
    ax_pr.set_title(f"{name}\nPrecision-Recall", fontsize=9)
    ax_pr.legend(fontsize=7)

    return metrics


def measure_time(model, X_train, y_train, X_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    model.predict(X_test)
    pred_time = time.perf_counter() - start

    return train_time, pred_time



models = {
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

fig = plt.figure(figsize=(20, 42))
gs = gridspec.GridSpec(11, 6, figure=fig, hspace=0.9, wspace=0.6)

times = []
all_metrics = []

#  Rows 0-3: per-model CM / ROC / PR 
for i, (name, model) in enumerate(models.items()):
    train_time, pred_time = measure_time(model, X_train, y_train, X_test)
    times.append((name, train_time, pred_time))

    ax_cm = fig.add_subplot(gs[i, 0:2])
    ax_roc = fig.add_subplot(gs[i, 2:4])
    ax_pr = fig.add_subplot(gs[i, 4:6])

    metrics = evaluate(name, model, X_test, y_test, ax_cm, ax_roc, ax_pr)
    all_metrics.append(metrics)

#  Row 4: classifier comparison bar chart 
metrics_df = pd.DataFrame(all_metrics).set_index("Model")
ax = fig.add_subplot(gs[4, :])
metrics_df.plot(kind="bar", ax=ax)
ax.set_title("Classifier Comparison")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right", fontsize=8)
ax.tick_params(axis="x", rotation=15)

#  Row 5: KNN accuracy vs k 
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    pred = knn_k.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

ax = fig.add_subplot(gs[5, :])
ax.plot(k_values, accuracies, marker="o")
ax.set_xticks(list(k_values))
ax.set_xlabel("k")
ax.set_ylabel("Accuracy")
ax.set_title("KNN Accuracy vs k")
ax.grid(True)

best_k = k_values[accuracies.index(max(accuracies))]
print("\nBest k:", best_k)

#  Row 6: GridSearchCV heatmaps per algorithm 
params = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "kd_tree", "ball_tree"]
}

start = time.perf_counter()
grid = GridSearchCV(
    KNeighborsClassifier(),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)
grid.fit(X_train, y_train)
grid_time = time.perf_counter() - start

print("\nGrid Search")
print("Best Parameters:", grid.best_params_)
print("Best Accuracy :", grid.best_score_)
print("Execution Time:", grid_time)

grid_df = pd.DataFrame(grid.cv_results_)
algorithms = params["algorithm"]

for i, algo in enumerate(algorithms):
    ax = fig.add_subplot(gs[6, i * 2:i * 2 + 2])
    subset = grid_df[grid_df["param_algorithm"] == algo]
    pivot = subset.pivot_table(
        index="param_n_neighbors",
        columns="param_weights",
        values="mean_test_score"
    )
    sns.heatmap(pivot, annot=True, cmap="viridis", ax=ax, cbar=False)
    ax.set_title(f"GridSearchCV: algorithm = {algo}", fontsize=9)

#  Row 7: RandomizedSearchCV score distribution 
start = time.perf_counter()
rand = RandomizedSearchCV(
    KNeighborsClassifier(),
    params,
    n_iter=10,
    cv=5,
    random_state=42,
    scoring="accuracy",
    n_jobs=-1
)
rand.fit(X_train, y_train)
rand_time = time.perf_counter() - start

print("\nRandomized Search")
print("Best Parameters:", rand.best_params_)
print("Best Accuracy :", rand.best_score_)
print("Execution Time:", rand_time)

scores = rand.cv_results_["mean_test_score"]

ax = fig.add_subplot(gs[7, :])
ax.hist(scores, bins=10)
ax.set_title("RandomizedSearchCV Score Distribution")
ax.set_xlabel("Accuracy")
ax.set_ylabel("Frequency")

#  Row 8: KDTree vs BallTree 
print("\nKDTree vs BallTree")
kdtree_balltree_results = []

for algo in ["kd_tree", "ball_tree"]:
    model = KNeighborsClassifier(n_neighbors=best_k, algorithm=algo)
    train_time, pred_time = measure_time(model, X_train, y_train, X_test)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    kdtree_balltree_results.append((algo, acc, train_time, pred_time))

    print("\nAlgorithm:", algo)
    print("Accuracy  :", acc)
    print("Training  :", train_time)
    print("Prediction:", pred_time)

kb_df = pd.DataFrame(
    kdtree_balltree_results,
    columns=["Algorithm", "Accuracy", "Training Time", "Prediction Time"]
)

ax = fig.add_subplot(gs[8, 0:2])
ax.bar(kb_df["Algorithm"], kb_df["Accuracy"])
ax.set_title("KD/Ball Tree: Accuracy")

ax = fig.add_subplot(gs[8, 2:4])
ax.bar(kb_df["Algorithm"], kb_df["Training Time"])
ax.set_title("KD/Ball Tree: Training Time (s)")

ax = fig.add_subplot(gs[8, 4:6])
ax.bar(kb_df["Algorithm"], kb_df["Prediction Time"])
ax.set_title("KD/Ball Tree: Prediction Time (s)")

# Row 9: 5-fold cross validation 
print("\n5-Fold Cross Validation")

cv_scores = {}

for name, model in models.items():
    pipe = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", model)
    ])
    scores = cross_val_score(pipe, X_all_raw, y_all, cv=5, scoring="accuracy")
    cv_scores[name] = scores

    print("\n", name)
    print(scores)
    print("Average:", scores.mean())

ax = fig.add_subplot(gs[9, :])
ax.bar(cv_scores.keys(), [s.mean() for s in cv_scores.values()])
ax.set_ylabel("Accuracy")
ax.set_title("5-Fold Cross Validation Accuracy (leakage-free)")
ax.tick_params(axis="x", rotation=15)

#  Row 10: training vs prediction time 
time_df = pd.DataFrame(times, columns=["Model", "Training", "Prediction"])

print("\nTraining / Prediction Time")
print(time_df)

ax = fig.add_subplot(gs[10, 0:3])
ax.bar(time_df["Model"], time_df["Training"])
ax.set_title("Training Time")
ax.set_ylabel("Seconds")
ax.tick_params(axis="x", rotation=15)

ax = fig.add_subplot(gs[10, 3:6])
ax.bar(time_df["Model"], time_df["Prediction"])
ax.set_title("Prediction Time")
ax.set_ylabel("Seconds")
ax.tick_params(axis="x", rotation=15)

fig.suptitle("Spambase Classification -- Full Results Dashboard", fontsize=18, y=1.001)
savefig(fig, "full_dashboard.png")
plt.show()