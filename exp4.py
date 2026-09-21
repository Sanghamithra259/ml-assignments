from eda_function import perform_eda
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

sns.set_style("whitegrid")


DATA_PATH = "spambase_csv.csv"   
TARGET_COL = "class"             # 1 = spam, 0 = ham
RANDOM_STATE = 42
LR_MAX_ITER = 20000               

# 1. LOAD DATA + EDA

df = perform_eda(DATA_PATH, target_col=TARGET_COL)

y = df[TARGET_COL]
X = df.drop(columns=[TARGET_COL])
feature_names = X.columns.tolist()

# 2. TRAIN / TEST SPLIT + STANDARDIZATION

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# 3. BASELINE LOGISTIC REGRESSION
start = time.time()
baseline_lr = LogisticRegression(max_iter=LR_MAX_ITER)
baseline_lr.fit(X_train_scaled, y_train)
baseline_lr_time = time.time() - start
baseline_lr_pred = baseline_lr.predict(X_test_scaled)

print("\n" + "=" * 60)
print("Baseline Logistic Regression (no tuning)")
print("=" * 60)
print(f"Accuracy : {accuracy_score(y_test, baseline_lr_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, baseline_lr_pred):.4f}")
print(f"Training Time: {baseline_lr_time:.4f}s")


# 4. HYPERPARAMETER TUNING - LOGISTIC REGRESSION

lr_param_grid = [
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["liblinear"]},
    {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100], "solver": ["saga"]},
]

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=LR_MAX_ITER),
    lr_param_grid,
    cv=kfold,
    scoring="accuracy",
    n_jobs=1,
    verbose=1,
)
lr_grid.fit(X_train_scaled, y_train)
best_lr = lr_grid.best_estimator_

print("\n" + "=" * 60)
print("Logistic Regression - Best Hyperparameters")
print("=" * 60)
print(lr_grid.best_params_)
print(f"Best CV Accuracy: {lr_grid.best_score_:.4f}")


# 5. SVM WITH DIFFERENT KERNELS (baseline, one per kernel)
kernels = ["linear", "poly", "rbf", "sigmoid"]
svm_kernel_results = []
fitted_svms = {}

for kernel in kernels:
    start = time.time()
    svm = SVC(kernel=kernel, gamma="scale", degree=3, probability=False)
    svm.fit(X_train_scaled, y_train)
    train_time = time.time() - start

    pred = svm.predict(X_test_scaled)
    fitted_svms[kernel] = svm

    svm_kernel_results.append({
        "Kernel": kernel,
        "Accuracy": accuracy_score(y_test, pred),
        "F1 Score": f1_score(y_test, pred),
        "Training Time (s)": train_time,
    })

svm_kernel_df = pd.DataFrame(svm_kernel_results)
print("\n" + "=" * 60)
print("SVM Kernel-wise Performance")
print("=" * 60)
print(svm_kernel_df.to_string(index=False))


# 6. HYPERPARAMETER TUNING - SVM

svm_param_grids = {
    "linear": {"kernel": ["linear"], "C": [0.1, 1, 10, 100]},
    "poly": {"kernel": ["poly"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"], "degree": [2, 3, 4]},
    "rbf": {"kernel": ["rbf"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
    "sigmoid": {"kernel": ["sigmoid"], "C": [0.1, 1, 10, 100], "gamma": ["scale", "auto"]},
}

svm_tuning_results = []
best_svms = {}

for kernel, grid_params in svm_param_grids.items():
    grid = GridSearchCV(SVC(), grid_params, cv=kfold, scoring="accuracy", n_jobs=1, verbose=1)
    grid.fit(X_train_scaled, y_train)
    best_svms[kernel] = grid.best_estimator_
    svm_tuning_results.append({
        "Kernel": kernel,
        "Best Parameters": grid.best_params_,
        "Best CV Accuracy": round(grid.best_score_, 4),
    })

svm_tuning_df = pd.DataFrame(svm_tuning_results)
print("\n" + "=" * 60)
print("SVM - Best Hyperparameters per Kernel")
print("=" * 60)
print(svm_tuning_df.to_string(index=False))

# overall best SVM = kernel with highest best_score_
best_svm_kernel = svm_tuning_df.sort_values("Best CV Accuracy", ascending=False).iloc[0]["Kernel"]
best_svm = best_svms[best_svm_kernel]

print(f"\nOverall best SVM kernel: {best_svm_kernel}")


# 7. TABLE 1 - HYPERPARAMETER TUNING SUMMARY (LR vs best SVM)
tuning_summary = pd.DataFrame([
    {"Model": "Logistic Regression", "Search Method": "Grid Search",
     "Best Parameters": lr_grid.best_params_, "Best CV Accuracy": round(lr_grid.best_score_, 4)},
    {"Model": "SVM", "Search Method": "Grid Search",
     "Best Parameters": best_svms[best_svm_kernel].get_params(),
     "Best CV Accuracy": svm_tuning_df.set_index("Kernel").loc[best_svm_kernel, "Best CV Accuracy"]},
])
print("\n" + "=" * 60)
print("Table: Hyperparameter Tuning Results")
print("=" * 60)
print(tuning_summary[["Model", "Search Method", "Best CV Accuracy"]].to_string(index=False))


# 8. FINAL TEST-SET EVALUATION (tuned LR vs tuned best SVM)
def evaluate(model, X_tr, y_tr, X_te, y_te):
    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start
    pred = model.predict(X_te)
    return {
        "Accuracy": accuracy_score(y_te, pred),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1 Score": f1_score(y_te, pred),
        "Training Time (s)": train_time,
    }, pred


lr_metrics, lr_test_pred = evaluate(best_lr, X_train_scaled, y_train, X_test_scaled, y_test)
svm_metrics, svm_test_pred = evaluate(best_svm, X_train_scaled, y_train, X_test_scaled, y_test)

print("\n" + "=" * 60)
print("Table: Logistic Regression Performance (tuned)")
print("=" * 60)
for k, v in lr_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n" + "=" * 60)
print(f"Table: Best SVM Performance (tuned, kernel = {best_svm_kernel})")
print("=" * 60)
for k, v in svm_metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


# 9. K-FOLD CROSS-VALIDATION (K = 5) - Table
lr_fold_scores = cross_val_score(best_lr, X_train_scaled, y_train, cv=kfold, scoring="accuracy")
svm_fold_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=kfold, scoring="accuracy")

cv_table = pd.DataFrame({
    "Fold": [f"Fold {i+1}" for i in range(5)],
    "Logistic Regression": lr_fold_scores,
    "SVM": svm_fold_scores,
})
avg_row = pd.DataFrame({
    "Fold": ["Average"],
    "Logistic Regression": [lr_fold_scores.mean()],
    "SVM": [svm_fold_scores.mean()],
})
cv_table = pd.concat([cv_table, avg_row], ignore_index=True)

print("\n" + "=" * 60)
print("Table: K-Fold Cross-Validation Results (K = 5)")
print("=" * 60)
print(cv_table.to_string(index=False))


# 10. VISUALIZATIONS 
from sklearn.metrics import confusion_matrix, roc_curve, auc

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes = axes.flatten()

# (a) Confusion matrix - Logistic Regression
cm_lr = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Confusion Matrix - Logistic Regression")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

# (b) Confusion matrix - best SVM
cm_svm = confusion_matrix(y_test, svm_test_pred)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", ax=axes[1])
axes[1].set_title(f"Confusion Matrix - SVM ({best_svm_kernel})")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# (c) ROC curves
for name, model in [("Logistic Regression", best_lr), ("SVM", best_svm)]:
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test_scaled)[:, 1]
    else:
        scores = model.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, scores)
    axes[2].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
axes[2].plot([0, 1], [0, 1], "k--")
axes[2].set_title("ROC Curves")
axes[2].set_xlabel("False Positive Rate")
axes[2].set_ylabel("True Positive Rate")
axes[2].legend()

# (d) SVM kernel comparison bar chart
axes[3].bar(svm_kernel_df["Kernel"], svm_kernel_df["Accuracy"], color="teal")
axes[3].set_title("SVM Accuracy by Kernel")
axes[3].set_ylabel("Accuracy")

# (e) K-Fold accuracy per fold (LR vs SVM)
fold_x = np.arange(5)
width = 0.35
axes[4].bar(fold_x - width / 2, lr_fold_scores, width, label="Logistic Regression")
axes[4].bar(fold_x + width / 2, svm_fold_scores, width, label="SVM")
axes[4].set_xticks(fold_x)
axes[4].set_xticklabels([f"Fold {i+1}" for i in range(5)])
axes[4].set_title("5-Fold CV Accuracy")
axes[4].set_ylabel("Accuracy")
axes[4].legend()

# (f) Logistic Regression coefficient magnitudes (top 10 features)
coef_series = pd.Series(best_lr.coef_[0], index=feature_names).abs().sort_values(ascending=False).head(10)
coef_series.plot(kind="bar", ax=axes[5], color="darkorange")
axes[5].set_title("Top 10 LR Coefficient Magnitudes")
axes[5].set_ylabel("|Coefficient|")
axes[5].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("exp4_all_visualizations.png", dpi=150)
plt.show()


# 11. COMPARATIVE ANALYSIS SUMMARY
print("\n" + "=" * 60)
print("Comparative Analysis")
print("=" * 60)
comparison = pd.DataFrame({
    "Criterion": ["Accuracy", "Training Time (s)", "Model Complexity", "Interpretability"],
    "Logistic Regression": [
        round(lr_metrics["Accuracy"], 4),
        round(lr_metrics["Training Time (s)"], 4),
        "Low",
        "High",
    ],
    "SVM": [
        round(svm_metrics["Accuracy"], 4),
        round(svm_metrics["Training Time (s)"], 4),
        "High",
        "Low",
    ],
})
print(comparison.to_string(index=False))