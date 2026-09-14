"""
Experiment 3: Regression Analysis using Linear and Regularized Models
ICS1512 - Machine Learning Algorithms Laboratory

Dataset : Loan Amount Prediction (train.csv, test.csv)
Models  : Linear Regression, Ridge, Lasso, Elastic Net

Fixes / speed-ups vs. the original draft:
  1. test.csv is now imputed (median for numeric, most-frequent for
     categorical, fit on train only) before modeling. Previously only
     train.csv passed through perform_eda's cleaning, so any missing
     value surviving in test.csv would crash model.predict() with
     "Input contains NaN".
  2. One StandardScaler ("full_scaler") is fit once on the full
     training set and reused consistently for X_train_full AND
     X_test. Previously the scaler was fit on an 80% split, used to
     scale X_test, then silently re-fit on the full set afterward --
     so final models and their test predictions were on two different
     scales.
  3. cross_validate() with multiple scorers replaces three separate
     cross_val_score() calls per model (MAE/MSE/R2 each refit the
     model from scratch) -> ~3x fewer redundant fits.
  4. n_jobs=-1 on GridSearchCV and cross_validate -> uses all CPU
     cores instead of one.
  5. sklearn.base.clone() replaces type(model)(**model.get_params())
     to rebuild estimators safely.
  6. Clear FileNotFoundError message if train.csv/test.csv aren't
     where expected, instead of a raw traceback.
"""

import os
import time
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

from eda_function import perform_eda

warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_style("whitegrid")

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET_COL = "Loan Sanction Amount (USD)"
ID_COLS = ["Customer ID", "Name", "Property ID"]
RANDOM_STATE = 42
N_JOBS = -1              
LASSO_MAX_ITER = 20000
LASSO_TOL = 1e-3         

for path in (TRAIN_PATH, TEST_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}' in the current working directory "
            f"({os.getcwd()}). Place train.csv / test.csv next to exp3.py, "
            f"or update TRAIN_PATH / TEST_PATH above."
        )

# 1. LOAD DATA + EDA
raw_train_df = pd.read_csv(TRAIN_PATH)
raw_train_df = raw_train_df.drop(columns=[c for c in ID_COLS if c in raw_train_df.columns])

TEMP_TRAIN_PATH = "_train_no_ids_temp.csv"
raw_train_df.to_csv(TEMP_TRAIN_PATH, index=False)


train_df = perform_eda(TEMP_TRAIN_PATH)
try:
    os.remove(TEMP_TRAIN_PATH)
except OSError:
    pass

test_df = pd.read_csv(TEST_PATH)
test_df = test_df.drop(columns=[c for c in ID_COLS if c in test_df.columns])

PLACEHOLDER_STRINGS = ["?", "NA", "N/A", "na", "n/a", "None", "none",
                        "Unknown", "unknown", "", " "]
test_df = test_df.replace(PLACEHOLDER_STRINGS, np.nan)


SENTINEL = -999
for frame in (train_df, test_df):
    for col in ("Property Price", TARGET_COL):
        if col in frame.columns:
            frame.loc[frame[col] == SENTINEL, col] = np.nan

train_df = train_df.dropna(subset=[TARGET_COL])
if "Property Price" in train_df.columns:
    train_df = train_df.dropna(subset=["Property Price"])

has_test_target = TARGET_COL in test_df.columns
if has_test_target:
    test_df = test_df.dropna(subset=[TARGET_COL])

# 2. PREPROCESSING (impute, encode categoricals, align, scale)
y_train_full = train_df[TARGET_COL]
X_train_full = train_df.drop(columns=[TARGET_COL])

y_test = test_df[TARGET_COL] if has_test_target else None
X_test = test_df.drop(columns=[TARGET_COL]) if has_test_target else test_df.copy()


numeric_cols = X_train_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train_full.select_dtypes(exclude=[np.number]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")


for col in numeric_cols:
    if col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

if numeric_cols:
    X_train_full[numeric_cols] = num_imputer.fit_transform(X_train_full[numeric_cols])
    if set(numeric_cols) <= set(X_test.columns):
        X_test[numeric_cols] = num_imputer.transform(X_test[numeric_cols])
    else:
        missing = set(numeric_cols) - set(X_test.columns)
        print(f"WARNING: test.csv is missing numeric column(s) {missing}; "
              f"skipping numeric imputation for test set.")

if categorical_cols:
    X_train_full[categorical_cols] = cat_imputer.fit_transform(X_train_full[categorical_cols])
    if set(categorical_cols) <= set(X_test.columns):
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

#  One-hot encode + align train/test to the same columns 
X_train_full = pd.get_dummies(X_train_full, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
X_train_full, X_test = X_train_full.align(X_test, join="left", axis=1, fill_value=0)

X_train_full = X_train_full.astype(float)
X_test = X_test.astype(float)

feature_names = X_train_full.columns.tolist()

# 80/20 split 
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE
)
split_scaler = StandardScaler()
X_train_scaled = split_scaler.fit_transform(X_train)
X_val_scaled = split_scaler.transform(X_val)


full_scaler = StandardScaler()
X_train_full_scaled = full_scaler.fit_transform(X_train_full)
X_test_scaled = full_scaler.transform(X_test)

kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 3. HYPERPARAMETER TUNING (Grid Search, 5-fold CV, R2 scoring)
param_grids = {
    "Ridge Regression": {"alpha": [0.01, 0.1, 1, 10, 100]},
    "Lasso Regression": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
    "Elastic Net Regression": {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8]},
}
base_models = {
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(max_iter=LASSO_MAX_ITER, tol=LASSO_TOL),
    "Elastic Net Regression": ElasticNet(max_iter=LASSO_MAX_ITER, tol=LASSO_TOL),
}

best_models = {}
tuning_results = []

for name, model in base_models.items():
    t0 = time.time()
    grid = GridSearchCV(model, param_grids[name], cv=kfold, scoring="r2", n_jobs=N_JOBS, verbose=1)
    grid.fit(X_train_full_scaled, y_train_full)
    best_models[name] = grid.best_estimator_
    tuning_results.append({
        "Model": name,
        "Search Method": "Grid Search",
        "Best Parameters": grid.best_params_,
        "Best CV R2": round(grid.best_score_, 4),
    })
    print(f"[{name}] tuned in {time.time() - t0:.1f}s -> "
          f"best R2={grid.best_score_:.4f}, params={grid.best_params_}")

best_models["Linear Regression"] = LinearRegression()

tuning_df = pd.DataFrame(tuning_results)
print("\n" + "=" * 60)
print("Table 1: Hyperparameter Tuning Summary")
print("=" * 60)
print(tuning_df.to_string(index=False))

# 4. CROSS-VALIDATION PERFORMANCE (K = 5) - Table 2
scoring = {
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2",
}

cv_results = []
for name, model in best_models.items():
    scores = cross_validate(model, X_train_full_scaled, y_train_full,
                             cv=kfold, scoring=scoring, n_jobs=N_JOBS)
    mae = -scores["test_MAE"].mean()
    mse = -scores["test_MSE"].mean()
    rmse = math.sqrt(mse)
    r2 = scores["test_R2"].mean()
    cv_results.append({"Model": name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2})

cv_df = pd.DataFrame(cv_results)
print("\n" + "=" * 60)
print("Table 2: Cross-Validation Performance (K = 5)")
print("=" * 60)
print(cv_df.to_string(index=False))

# 5. FIT FINAL MODELS + TEST SET PERFORMANCE - Table 3
fitted_models, train_errors, val_errors, training_times = {}, {}, {}, {}
test_results, test_preds = [], {}

for name, model in best_models.items():
    start = time.time()
    model_split = clone(model)
    model_split.fit(X_train_scaled, y_train)
    training_times[name] = time.time() - start

    train_errors[name] = mean_squared_error(y_train, model_split.predict(X_train_scaled))
    val_errors[name] = mean_squared_error(y_val, model_split.predict(X_val_scaled))

    final_model = clone(model)
    final_model.fit(X_train_full_scaled, y_train_full)
    fitted_models[name] = final_model

    pred = final_model.predict(X_test_scaled)
    test_preds[name] = pred

    if has_test_target:
        mse = mean_squared_error(y_test, pred)
        test_results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "R2": r2_score(y_test, pred),
        })

if has_test_target:
    test_df_results = pd.DataFrame(test_results)
    print("\n" + "=" * 60)
    print("Table 3: Test Set Performance")
    print("=" * 60)
    print(test_df_results.to_string(index=False))
else:
    print("\ntest.csv has no target column -> Table 3 (test metrics) skipped; "
          "predictions were still generated for each model.")

# 6. COEFFICIENT COMPARISON - Table 4
coef_data = {"Feature": feature_names}
for name in ["Linear Regression", "Ridge Regression", "Lasso Regression", "Elastic Net Regression"]:
    coef_data[name] = fitted_models[name].coef_

coef_df = pd.DataFrame(coef_data)
print("\n" + "=" * 60)
print("Table 4: Coefficient Comparison")
print("=" * 60)
print(coef_df.to_string(index=False))

# 7. ALL REQUIRED VISUALIZATIONS IN ONE COMBINED FIGURE
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
axes = axes.flatten()

sns.histplot(y_train_full, bins=30, kde=True, ax=axes[0], color="steelblue")
axes[0].set_title("Target Variable Distribution")
axes[0].set_xlabel(TARGET_COL)

numeric_features = [c for c in feature_names if X_train_full[c].nunique() > 2]
if numeric_features:
    corrs = X_train_full[numeric_features].corrwith(y_train_full).abs().sort_values(ascending=False)
    top_feature = corrs.index[0]
    axes[1].scatter(X_train_full[top_feature], y_train_full, alpha=0.4, color="darkorange")
    axes[1].set_title(f"{top_feature} vs {TARGET_COL}")
    axes[1].set_xlabel(top_feature)
    axes[1].set_ylabel(TARGET_COL)

best_name = cv_df.sort_values("R2", ascending=False).iloc[0]["Model"]
val_pred_best = clone(best_models[best_name]).fit(X_train_scaled, y_train).predict(X_val_scaled)
axes[2].scatter(y_val, val_pred_best, alpha=0.4, color="seagreen")
lims = [min(y_val.min(), val_pred_best.min()), max(y_val.max(), val_pred_best.max())]
axes[2].plot(lims, lims, "r--")
axes[2].set_title(f"Predicted vs Actual ({best_name})")
axes[2].set_xlabel("Actual")
axes[2].set_ylabel("Predicted")

residuals = y_val.values - val_pred_best
axes[3].scatter(val_pred_best, residuals, alpha=0.4, color="crimson")
axes[3].axhline(0, color="black", linestyle="--")
axes[3].set_title(f"Residual Plot ({best_name})")
axes[3].set_xlabel("Predicted")
axes[3].set_ylabel("Residual")

model_names = list(train_errors.keys())
x = np.arange(len(model_names))
width = 0.35
axes[4].bar(x - width / 2, [train_errors[m] for m in model_names], width, label="Train MSE")
axes[4].bar(x + width / 2, [val_errors[m] for m in model_names], width, label="Validation MSE")
axes[4].set_xticks(x)
axes[4].set_xticklabels(model_names, rotation=20, ha="right")
axes[4].set_title("Training vs Validation Error")
axes[4].set_ylabel("MSE")
axes[4].legend()

plot_coef_df = coef_df.set_index("Feature")
top10 = plot_coef_df["Linear Regression"].abs().sort_values(ascending=False).head(10).index
plot_coef_df.loc[top10].plot(kind="bar", ax=axes[5])
axes[5].set_title("Coefficient Comparison (Top 10 Features)")
axes[5].set_ylabel("Coefficient Value")
axes[5].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("exp3_all_visualizations.png", dpi=150)
plt.show()

# 8. TRAINING TIME SUMMARY
print("\n" + "=" * 60)
print("Training Time (seconds)")
print("=" * 60)
for name, t in training_times.items():
    print(f"{name}: {t:.4f}s")
