import os
import math
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from eda_function import perform_eda

sns.set_style("whitegrid")
RANDOM_STATE = 42

# ======================================================================
# 0. CONFIG -- edit these two lines for your dataset
# ======================================================================
FILE_PATH = "spambase.csv"     # path to your CSV
TARGET_COL = "class"           # name of the target/label column
PLOTS_DIR = "plots_exp6"
os.makedirs(PLOTS_DIR, exist_ok=True)

# If the real dataset isn't found, fall back to a synthetic binary
# classification dataset so the pipeline can still be demoed / tested.
if not os.path.exists(FILE_PATH):
    print(f"'{FILE_PATH}' not found -> generating a synthetic dataset "
          f"so you can test-run the pipeline. Replace FILE_PATH/TARGET_COL "
          f"with your real dataset before submitting.")
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(
        n_samples=600, n_features=20, n_informative=12, n_redundant=4,
        n_classes=2, random_state=RANDOM_STATE
    )
    df_syn = pd.DataFrame(X_syn, columns=[f"f{i}" for i in range(X_syn.shape[1])])
    df_syn[TARGET_COL] = y_syn
    df_syn.to_csv(FILE_PATH, index=False)

# ======================================================================
# 1. Load & clean data using the provided EDA function
# ======================================================================
df = perform_eda(FILE_PATH, target_col=TARGET_COL, plots_dir=PLOTS_DIR, show=False)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

# Encode target if it's categorical/text
if y.dtype == object or str(y.dtype).startswith("category"):
    y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)

# Keep only numeric features for the models below. If you have
# categorical predictor columns, one-hot encode them before this point.
X = X.select_dtypes(include=np.number)

print("\nFinal feature matrix:", X.shape)
print("Target distribution:\n", y.value_counts())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ======================================================================
# 2. PCA design: scree plot + variance-target component count
#    (combined into a single figure: individual + cumulative variance)
# ======================================================================
scaler_full = StandardScaler()
X_scaled_full = scaler_full.fit_transform(X)

pca_full = PCA(random_state=RANDOM_STATE).fit(X_scaled_full)
explained = pca_full.explained_variance_ratio_
cum_var = np.cumsum(explained)

VARIANCE_TARGET = 0.95
n_components = int(np.argmax(cum_var >= VARIANCE_TARGET) + 1)
print(f"\nComponents needed for {VARIANCE_TARGET * 100:.0f}% variance: {n_components} "
      f"(out of {X.shape[1]} original features)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(range(1, len(explained) + 1), explained, marker="o")
axes[0].set_title("Scree Plot (Per-Component Variance)")
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Explained Variance Ratio")

axes[1].plot(range(1, len(cum_var) + 1), cum_var, marker="o", color="darkorange")
axes[1].axhline(VARIANCE_TARGET, color="red", linestyle="--",
                 label=f"{int(VARIANCE_TARGET * 100)}% threshold")
axes[1].axvline(n_components, color="green", linestyle="--",
                 label=f"{n_components} components")
axes[1].set_title("Cumulative Explained Variance")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Explained Variance")
axes[1].legend()

fig.suptitle("PCA Variance Analysis", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "pca_scree_combined.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 3. Models + hyperparameter grids
#    (keys use the "clf__" pipeline-step prefix)
# ======================================================================
base_models = {
    "SVM": (
        SVC(probability=True, random_state=RANDOM_STATE),
        {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.1], "clf__kernel": ["rbf", "linear"]},
    ),
    "Naive Bayes": (
        GaussianNB(),
        {"clf__var_smoothing": np.logspace(-9, -6, 4)},
    ),
    "KNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"],
         "clf__metric": ["euclidean", "manhattan"]},
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
    ),
    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {"clf__max_depth": [3, 5, 10, None], "clf__min_samples_split": [2, 5, 10]},
    ),
    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [50, 100, 200], "clf__learning_rate": [0.5, 1.0]},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
    ),
    "XGBoost": (
        XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5]},
    ),
}

stacking_estimators = [
    ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    ("svc", SVC(probability=True, random_state=RANDOM_STATE)),
    ("knn", KNeighborsClassifier()),
]
stacking_model = StackingClassifier(
    estimators=stacking_estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    cv=5,
)
stacking_grid = {"clf__final_estimator__C": [0.1, 1, 10]}

all_models = dict(base_models)
all_models["Stacking"] = (stacking_model, stacking_grid)


# ======================================================================
# 4. Train + tune + cross-validate one model, in one setting
# ======================================================================
def evaluate_model(name, estimator, param_grid, X, y, cv, use_pca, n_components):
    steps = [("scaler", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("clf", estimator))
    pipe = Pipeline(steps)

    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    grid.fit(X, y)
    best_pipe = grid.best_estimator_

    fold_acc = cross_val_score(best_pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    fold_f1 = cross_val_score(best_pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)

    y_pred = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=-1)
    try:
        y_proba = cross_val_predict(best_pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    except Exception:
        y_proba = None

    return {
        "model": name,
        "setting": "With-PCA" if use_pca else "No-PCA",
        "best_params": grid.best_params_,
        "fold_acc": fold_acc,
        "avg_accuracy": fold_acc.mean(),
        "std_accuracy": fold_acc.std(),
        "avg_f1": fold_f1.mean(),
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


# ======================================================================
# 5. Run every model in both settings
# ======================================================================
results = []
for name, (estimator, grid_params) in all_models.items():
    print(f"\n=== {name} ===")
    for use_pca in (False, True):
        res = evaluate_model(name, estimator, grid_params, X, y, cv,
                              use_pca=use_pca, n_components=n_components)
        print(f"{res['setting']:10s} | acc={res['avg_accuracy']:.4f} "
              f"(+/-{res['std_accuracy']:.4f}) | f1={res['avg_f1']:.4f} | "
              f"best_params={res['best_params']}")
        results.append(res)

# ======================================================================
# 6. Table 5: fold-wise + average results (No-PCA vs With-PCA)
# ======================================================================
summary_rows = []
for r in results:
    row = {"Model": r["model"], "Setting": r["setting"]}
    for i, s in enumerate(r["fold_acc"], start=1):
        row[f"Fold {i}"] = round(float(s), 4)
    row["Avg Accuracy"] = round(r["avg_accuracy"], 4)
    row["Avg F1"] = round(r["avg_f1"], 4)
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(PLOTS_DIR, "cv_results_summary.csv"), index=False)
print("\nCross-Validation Summary (Table 5)")
print(summary_df.to_string(index=False))

params_df = pd.DataFrame(
    [{"Model": r["model"], "Setting": r["setting"], "Best Params": r["best_params"]} for r in results]
)
params_df.to_csv(os.path.join(PLOTS_DIR, "best_hyperparameters.csv"), index=False)

# ======================================================================
# 7. Combined bar chart: Avg Accuracy & Avg F1, No-PCA vs With-PCA
#    (one figure, two related panels -- like the EDA grids)
# ======================================================================
pivot_acc = summary_df.pivot(index="Model", columns="Setting", values="Avg Accuracy")
pivot_f1 = summary_df.pivot(index="Model", columns="Setting", values="Avg F1")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
pivot_acc.plot(kind="bar", ax=axes[0])
axes[0].set_title("Average Accuracy: No-PCA vs With-PCA")
axes[0].set_ylabel("Accuracy")
axes[0].tick_params(axis="x", rotation=45)

pivot_f1.plot(kind="bar", ax=axes[1])
axes[1].set_title("Average F1-score: No-PCA vs With-PCA")
axes[1].set_ylabel("F1-score (weighted)")
axes[1].tick_params(axis="x", rotation=45)

fig.suptitle("Model Performance Comparison", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "performance_comparison_combined.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 8. Combined fold-wise stability boxplot (all models, both settings)
# ======================================================================
fold_records = []
for r in results:
    for i, s in enumerate(r["fold_acc"], start=1):
        fold_records.append({"Model": r["model"], "Setting": r["setting"], "Fold": i, "Accuracy": s})
fold_df = pd.DataFrame(fold_records)

fig, ax = plt.subplots(figsize=(16, 7))
sns.boxplot(data=fold_df, x="Model", y="Accuracy", hue="Setting", ax=ax)
ax.set_title("Fold-wise Accuracy Distribution (Stability): No-PCA vs With-PCA")
ax.tick_params(axis="x", rotation=45)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "foldwise_stability_combined.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 9. Combined ROC curves -- one subplot per model, both settings overlaid
# ======================================================================
model_names = list(all_models.keys())
cols = 3
rows = math.ceil(len(model_names) / cols)
fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
axes = np.array(axes).reshape(-1)

for i, name in enumerate(model_names):
    ax = axes[i]
    for r in results:
        if r["model"] == name and r["y_proba"] is not None:
            fpr, tpr, _ = roc_curve(y, r["y_proba"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{r['setting']} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("ROC Curves: No-PCA vs With-PCA (All Models)", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "roc_curves_combined.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 10. Combined confusion matrices -- one grid, all model/setting pairs
# ======================================================================
cols_cm = 4
rows_cm = math.ceil(len(results) / cols_cm)
fig, axes = plt.subplots(rows_cm, cols_cm, figsize=(5 * cols_cm, 4 * rows_cm))
axes = np.array(axes).reshape(-1)

for i, r in enumerate(results):
    cm = confusion_matrix(y, r["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[i], colorbar=False, cmap="Blues")
    axes[i].set_title(f"{r['model']} ({r['setting']})", fontsize=9)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Confusion Matrices: No-PCA vs With-PCA (All Models)", fontsize=16)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrices_combined.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ======================================================================
# 11. Observation helper: which models improved with PCA?
# ======================================================================
delta_rows = []
for name in model_names:
    no_pca = summary_df[(summary_df.Model == name) & (summary_df.Setting == "No-PCA")]["Avg Accuracy"].values[0]
    with_pca = summary_df[(summary_df.Model == name) & (summary_df.Setting == "With-PCA")]["Avg Accuracy"].values[0]
    delta_rows.append({"Model": name, "No-PCA Acc": no_pca, "With-PCA Acc": with_pca,
                        "Delta (With - No)": round(with_pca - no_pca, 4)})
delta_df = pd.DataFrame(delta_rows).sort_values("Delta (With - No)", ascending=False)
delta_df.to_csv(os.path.join(PLOTS_DIR, "pca_delta_by_model.csv"), index=False)
print("\nPCA Impact by Model (sorted, best improvement first)")
print(delta_df.to_string(index=False))
