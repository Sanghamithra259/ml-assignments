import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


import pandas as pd
from eda_function import perform_eda

columns = [
    "id",
    "diagnosis",

    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# Load original UCI dataset
df = pd.read_csv(
    "wdbc.data",
    header=None,
    names=columns
)

# Save it with headers
df.to_csv(
    "wdbc.csv",
    index=False
)

# Now use your EDA function
df = perform_eda(
    "wdbc.csv",
    drop_missing=False,
    remove_duplicates=True,
    remove_outliers=False
)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# 2. SAVE TEMPORARY CSV FOR YOUR EDA FUNCTION

eda_file = "wdbc_with_headers.csv"

df.to_csv(
    eda_file,
    index=False
)


# 3. PERFORM EDA


df = perform_eda(
    eda_file,
    drop_missing=False,
    remove_duplicates=True,
    remove_outliers=False
)


# 4. CLASS DISTRIBUTION

print("\nCLASS DISTRIBUTION")
print(
    df["diagnosis"].value_counts()
)


plt.figure(figsize=(6, 4))

sns.countplot(
    x="diagnosis",
    data=df
)

plt.title(
    "Diagnosis Class Distribution"
)

plt.xlabel(
    "Diagnosis"
)

plt.ylabel(
    "Number of Samples"
)

plt.show()


# 5. ENCODE TARGET

df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})


print("\nEncoded target:")
print(
    df["diagnosis"].value_counts()
)

# 6. SEPARATE FEATURES AND TARGET

X = df.drop(
    columns=["id", "diagnosis"]
)

y = df["diagnosis"]


print("\nFeature shape:", X.shape)
print("Target shape :", y.shape)


# 7. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# 8. 5-FOLD STRATIFIED CROSS VALIDATION

cv = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)


# 9. DECISION TREE

dt = DecisionTreeClassifier(
    random_state=42
)


dt_param_grid = {

    "criterion": [
        "gini",
        "entropy"
    ],

    "max_depth": [
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        None
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ]
}


dt_grid = GridSearchCV(

    estimator=dt,

    param_grid=dt_param_grid,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1,

    return_train_score=True
)


dt_grid.fit(
    X_train,
    y_train
)


print("\n==============================")
print("DECISION TREE")
print("==============================")

print(
    "\nBest Parameters:"
)

print(
    dt_grid.best_params_
)

print(
    "\nBest CV Accuracy:",
    dt_grid.best_score_
)


# 10. DECISION TREE CV TABLE

dt_results = pd.DataFrame(
    dt_grid.cv_results_
)


dt_results = dt_results.sort_values(
    by="mean_test_score",
    ascending=False
)


print(
    "\nTop Decision Tree Results:"
)

print(
    dt_results[
        [
            "param_criterion",
            "param_max_depth",
            "param_min_samples_split",
            "param_min_samples_leaf",
            "mean_test_score"
        ]
    ].head(10)
)


# 11. BEST DECISION TREE

best_dt = dt_grid.best_estimator_


best_dt.fit(
    X_train,
    y_train
)


dt_pred = best_dt.predict(
    X_test
)


dt_prob = best_dt.predict_proba(
    X_test
)[:, 1]


# 12. DECISION TREE METRICS

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

dt_precision = precision_score(
    y_test,
    dt_pred
)

dt_recall = recall_score(
    y_test,
    dt_pred
)

dt_f1 = f1_score(
    y_test,
    dt_pred
)

dt_auc = roc_auc_score(
    y_test,
    dt_prob
)


print(
    "\nDecision Tree Test Metrics"
)

print(
    "Accuracy :",
    dt_accuracy
)

print(
    "Precision:",
    dt_precision
)

print(
    "Recall   :",
    dt_recall
)

print(
    "F1 Score :",
    dt_f1
)

print(
    "AUC      :",
    dt_auc
)


# 13. DECISION TREE CONFUSION MATRIX

dt_cm = confusion_matrix(
    y_test,
    dt_pred
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    dt_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Benign",
        "Malignant"
    ],
    yticklabels=[
        "Benign",
        "Malignant"
    ]
)

plt.title(
    "Decision Tree Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


# 14. RANDOM FOREST

rf = RandomForestClassifier(
    random_state=42
)


rf_param_grid = {

    "n_estimators": [
        50,
        100,
        150
    ],

    "max_depth": [
        None,
        3,
        5,
        8,
        10
    ],

    "max_features": [
        "sqrt",
        "log2"
    ],

    "bootstrap": [
        True,
        False
    ]
}


rf_grid = GridSearchCV(

    estimator=rf,

    param_grid=rf_param_grid,

    cv=cv,

    scoring="accuracy",

    n_jobs=-1,

    return_train_score=True
)


rf_grid.fit(
    X_train,
    y_train
)


print("\n==============================")
print("RANDOM FOREST")
print("==============================")


print(
    "\nBest Parameters:"
)

print(
    rf_grid.best_params_
)


print(
    "\nBest CV Accuracy:",
    rf_grid.best_score_
)


# 15. RANDOM FOREST CV TABLE

rf_results = pd.DataFrame(
    rf_grid.cv_results_
)


rf_results = rf_results.sort_values(
    by="mean_test_score",
    ascending=False
)


print(
    "\nTop Random Forest Results:"
)

print(
    rf_results[
        [
            "param_n_estimators",
            "param_max_depth",
            "param_max_features",
            "param_bootstrap",
            "mean_test_score"
        ]
    ].head(10)
)

# 16. BEST RANDOM FOREST

best_rf = rf_grid.best_estimator_


best_rf.fit(
    X_train,
    y_train
)


rf_pred = best_rf.predict(
    X_test
)


rf_prob = best_rf.predict_proba(
    X_test
)[:, 1]


# 17. RANDOM FOREST METRICS

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_f1 = f1_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)


print(
    "\nRandom Forest Test Metrics"
)

print(
    "Accuracy :",
    rf_accuracy
)

print(
    "Precision:",
    rf_precision
)

print(
    "Recall   :",
    rf_recall
)

print(
    "F1 Score :",
    rf_f1
)

print(
    "AUC      :",
    rf_auc
)


# 18. RANDOM FOREST CONFUSION MATRIX

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)


plt.figure(figsize=(6, 5))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=[
        "Benign",
        "Malignant"
    ],
    yticklabels=[
        "Benign",
        "Malignant"
    ]
)

plt.title(
    "Random Forest Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.show()


# 19. FOLD-WISE COMPARISON

dt_cv = cross_validate(

    best_dt,

    X_train,

    y_train,

    cv=cv,

    scoring=[
        "accuracy",
        "f1"
    ]
)


rf_cv = cross_validate(

    best_rf,

    X_train,

    y_train,

    cv=cv,

    scoring=[
        "accuracy",
        "f1"
    ]
)


comparison = pd.DataFrame({

    "Fold": [
        1, 2, 3, 4, 5
    ],

    "Decision Tree Accuracy":
        dt_cv["test_accuracy"],

    "Random Forest Accuracy":
        rf_cv["test_accuracy"],

    "Decision Tree F1":
        dt_cv["test_f1"],

    "Random Forest F1":
        rf_cv["test_f1"]
})


print(
    "\n=============================="
)

print(
    "5-FOLD CV COMPARISON"
)

print(
    "=============================="
)

print(
    comparison
)


print(
    "\nAverage Decision Tree Accuracy:",
    comparison[
        "Decision Tree Accuracy"
    ].mean()
)


print(
    "Average Random Forest Accuracy:",
    comparison[
        "Random Forest Accuracy"
    ].mean()
)


print(
    "\nAverage Decision Tree F1:",
    comparison[
        "Decision Tree F1"
    ].mean()
)


print(
    "Average Random Forest F1:",
    comparison[
        "Random Forest F1"
    ].mean()
)


# 20. FINAL MODEL COMPARISON

performance = pd.DataFrame({

    "Model": [
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ],

    "Precision": [
        dt_precision,
        rf_precision
    ],

    "Recall": [
        dt_recall,
        rf_recall
    ],

    "F1 Score": [
        dt_f1,
        rf_f1
    ],

    "AUC": [
        dt_auc,
        rf_auc
    ]
})


print(
    "\n=============================="
)

print(
    "FINAL COMPARISON"
)

print(
    "=============================="
)

print(
    performance
)


# 21. ROC CURVE

dt_fpr, dt_tpr, _ = roc_curve(
    y_test,
    dt_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)


plt.figure(figsize=(8, 6))


plt.plot(
    dt_fpr,
    dt_tpr,
    label=f"Decision Tree AUC={dt_auc:.3f}"
)


plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest AUC={rf_auc:.3f}"
)


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)


plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve Comparison"
)

plt.legend()

plt.show()
