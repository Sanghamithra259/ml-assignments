import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from eda_function import perform_eda

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


# ============================================================
# 1. LOAD AND PERFORM EDA
# ============================================================

file_path = "wdbc.csv"

df = perform_eda(
    file_path,
    target_col="diagnosis",
    plots_dir="plots",
    show=False
)

print("\nDataset after EDA:")
print(df.head())

print("\nFinal Dataset Shape:", df.shape)


# ============================================================
# 2. PREPROCESS DATA
# ============================================================

# Remove ID column because it is not a useful feature
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Convert diagnosis:
# B = 0
# M = 1
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

# Separate features and target
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nNumber of Features:", X.shape[1])

print("\nTarget Distribution:")
print(y.value_counts())


# ============================================================
# 3. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set:", X_train.shape)
print("Testing Set :", X_test.shape)


# ============================================================
# 4. CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 5. BAGGING CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("BAGGING CLASSIFIER")
print("=" * 70)

bagging_base = DecisionTreeClassifier(
    random_state=42
)

bagging = BaggingClassifier(
    estimator=bagging_base,
    random_state=42
)

bagging_params = {
    "n_estimators": [3, 5, 10, 20],
    "max_samples": [0.5, 0.8, 1.0],
    "max_features": [0.5, 0.8, 1.0]
}

bagging_grid = GridSearchCV(
    estimator=bagging,
    param_grid=bagging_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=False
)

bagging_grid.fit(X_train, y_train)

best_bagging = bagging_grid.best_estimator_

print("\nBest Bagging Parameters:")
print(bagging_grid.best_params_)

print(
    "\nBest Bagging CV Accuracy: "
    f"{bagging_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# TABLE 1 - BAGGING HYPERPARAMETER EVALUATION
# ============================================================

bagging_results = pd.DataFrame(
    bagging_grid.cv_results_
)

bagging_table = bagging_results[
    [
        "param_n_estimators",
        "param_max_samples",
        "param_max_features",
        "mean_test_score"
    ]
].copy()

bagging_table.columns = [
    "n_estimators",
    "max_samples",
    "max_features",
    "Avg CV Accuracy (%)"
]

bagging_table["Avg CV Accuracy (%)"] = (
    bagging_table["Avg CV Accuracy (%)"] * 100
).round(2)

print("\nTable 1: Bagging Hyperparameter Evaluation")
print(bagging_table.to_string(index=False))


# ============================================================
# 6. ADABOOST CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("ADABOOST CLASSIFIER")
print("=" * 70)

adaboost = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    random_state=42
)

adaboost_params = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2, 0.5]
}

adaboost_grid = GridSearchCV(
    estimator=adaboost,
    param_grid=adaboost_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=False
)

adaboost_grid.fit(X_train, y_train)

best_adaboost = adaboost_grid.best_estimator_

print("\nBest AdaBoost Parameters:")
print(adaboost_grid.best_params_)

print(
    "\nBest AdaBoost CV Accuracy: "
    f"{adaboost_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# 7. GRADIENT BOOSTING CLASSIFIER
# ============================================================

print("\n")
print("=" * 70)
print("GRADIENT BOOSTING CLASSIFIER")
print("=" * 70)

gradient_boosting = GradientBoostingClassifier(
    random_state=42
)

gradient_params = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [1, 2, 3]
}

gradient_grid = GridSearchCV(
    estimator=gradient_boosting,
    param_grid=gradient_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=False
)

gradient_grid.fit(X_train, y_train)

best_gradient = gradient_grid.best_estimator_

print("\nBest Gradient Boosting Parameters:")
print(gradient_grid.best_params_)

print(
    "\nBest Gradient Boosting CV Accuracy: "
    f"{gradient_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# TABLE 2 - BOOSTING HYPERPARAMETER EVALUATION
# ============================================================

gradient_results = pd.DataFrame(
    gradient_grid.cv_results_
)

gradient_table = gradient_results[
    [
        "param_n_estimators",
        "param_learning_rate",
        "param_max_depth",
        "mean_test_score"
    ]
].copy()

gradient_table.columns = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "Avg CV Accuracy (%)"
]

gradient_table["Avg CV Accuracy (%)"] = (
    gradient_table["Avg CV Accuracy (%)"] * 100
).round(2)

print("\nTable 2: Boosting Hyperparameter Evaluation")
print(gradient_table.to_string(index=False))


# ============================================================
# 8. STACKED ENSEMBLE
# ============================================================

print("\n")
print("=" * 70)
print("STACKED ENSEMBLE")
print("=" * 70)


# SVM with scaling
svm_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        random_state=42
    ))
])


# Naive Bayes
nb_model = GaussianNB()


# Decision Tree
dt_model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)


# Base learners
base_models = [
    ("svm", svm_model),
    ("nb", nb_model),
    ("dt", dt_model)
]


# Meta learner
meta_learner = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,
    stack_method="auto",
    n_jobs=-1
)


# Hyperparameter search for meta learner
stacking_params = {
    "final_estimator__logistic__C": [0.1, 1.0, 10.0]
}


stacking_grid = GridSearchCV(
    estimator=stacking,
    param_grid=stacking_params,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    return_train_score=False
)

stacking_grid.fit(X_train, y_train)

best_stacking = stacking_grid.best_estimator_

print("\nBest Stacking Parameters:")
print(stacking_grid.best_params_)

print(
    "\nBest Stacking CV Accuracy: "
    f"{stacking_grid.best_score_ * 100:.2f}%"
)


# ============================================================
# TABLE 3 - STACKING EVALUATION
# ============================================================

stacking_results = pd.DataFrame(
    stacking_grid.cv_results_
)

stacking_table = stacking_results[
    [
        "param_final_estimator__logistic__C",
        "mean_test_score"
    ]
].copy()

stacking_table.columns = [
    "Meta Learner C",
    "Avg CV Accuracy (%)"
]

stacking_table["Avg CV Accuracy (%)"] = (
    stacking_table["Avg CV Accuracy (%)"] * 100
).round(2)

print("\nTable 3: Stacked Ensemble Evaluation")

print("\nBase Models:")
print("SVM + Naive Bayes + Decision Tree")

print("\nMeta Learner:")
print("Logistic Regression")

print("\nHyperparameter Results:")
print(stacking_table.to_string(index=False))


# ============================================================
# 9. FUNCTION FOR MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    y_pred = model.predict(X_test)

    # Probability / decision score for ROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]

    else:
        y_score = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        y_score
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print("\n")
    print("-" * 60)
    print(model_name)
    print("-" * 60)

    print(f"Accuracy  : {accuracy * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%")
    print(f"AUC       : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc,
        "y_score": y_score
    }


# ============================================================
# 10. TRAIN / EVALUATE MODELS
# ============================================================

bagging_result = evaluate_model(
    best_bagging,
    X_test,
    y_test,
    "Bagging"
)

adaboost_result = evaluate_model(
    best_adaboost,
    X_test,
    y_test,
    "AdaBoost"
)

gradient_result = evaluate_model(
    best_gradient,
    X_test,
    y_test,
    "Gradient Boosting"
)

stacking_result = evaluate_model(
    best_stacking,
    X_test,
    y_test,
    "Stacked Ensemble"
)


# ============================================================
# 11. TABLE 4 - PERFORMANCE COMPARISON
# ============================================================

results = [
    bagging_result,
    adaboost_result,
    gradient_result,
    stacking_result
]

comparison_table = pd.DataFrame([
    {
        "Model": result["Model"],
        "Accuracy (%)": result["Accuracy"] * 100,
        "Precision (%)": result["Precision"] * 100,
        "Recall (%)": result["Recall"] * 100,
        "F1 Score (%)": result["F1 Score"] * 100,
        "AUC": result["AUC"]
    }
    for result in results
])

comparison_table[
    [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "AUC"
    ]
] = comparison_table[
    [
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "AUC"
    ]
].round(2)

print("\n")
print("=" * 70)
print("TABLE 4: PERFORMANCE COMPARISON OF ENSEMBLE MODELS")
print("=" * 70)

print(comparison_table.to_string(index=False))


# ============================================================
# 12. SAVE PERFORMANCE TABLE
# ============================================================

comparison_table.to_csv(
    "ensemble_performance_comparison.csv",
    index=False
)


# ============================================================
# 13. ROC CURVE
# ============================================================

plt.figure(figsize=(9, 7))

for result in results:

    fpr, tpr, _ = roc_curve(
        y_test,
        result["y_score"]
    )

    plt.plot(
        fpr,
        tpr,
        label=f'{result["Model"]} (AUC = {result["AUC"]:.4f})'
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve - Ensemble Models")

plt.legend()

plt.tight_layout()

plt.savefig(
    "plots/ensemble_roc_curve.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 14. FINAL RESULT
# ============================================================

best_model = comparison_table.loc[
    comparison_table["Accuracy (%)"].idxmax()
]

print("\n")
print("=" * 70)
print("BEST PERFORMING MODEL")
print("=" * 70)

print(
    f"Model    : {best_model['Model']}"
)

print(
    f"Accuracy : {best_model['Accuracy (%)']:.2f}%"
)

print(
    f"F1 Score : {best_model['F1 Score (%)']:.2f}%"
)

print(
    f"AUC      : {best_model['AUC']:.4f}"
)
