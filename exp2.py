from eda_function import perform_eda
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.preprocessing import MinMaxScaler
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
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Data
df = perform_eda("spam.csv")

X = df.drop("class", axis=1)
y = df["class"]

# Convert labels if needed
if y.dtype == object:
    y = y.map({"ham": 0, "spam": 1})

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Helpers
def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Precision:", precision_score(y_test, pred))
    print("Recall   :", recall_score(y_test, pred))
    print("F1 Score :", f1_score(y_test, pred))
    print("ROC AUC  :", roc_auc_score(y_test, prob))

    ConfusionMatrixDisplay.from_predictions(y_test, pred)
    plt.title(name + " Confusion Matrix")
    plt.show()

    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(name + " ROC Curve")
    plt.show()

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
    plt.title(name + " Precision-Recall Curve")
    plt.show()


def measure_time(model, X_train, y_train, X_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    start = time.perf_counter()
    model.predict(X_test)
    pred_time = time.perf_counter() - start

    return train_time, pred_time


# Models
models = {
    "Gaussian NB": GaussianNB(),
    "Multinomial NB": MultinomialNB(),
    "Bernoulli NB": BernoulliNB(),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

times = []

# Evaluation
for name, model in models.items():
    train_time, pred_time = measure_time(
        model,
        X_train,
        y_train,
        X_test
    )

    times.append((name, train_time, pred_time))
    evaluate(name, model, X_test, y_test)

# KNN tuning
k_values = range(1, 21)
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, pred))

plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker="o")
plt.xticks(list(k_values))
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy vs k")
plt.grid(True)
plt.show()

best_k = k_values[accuracies.index(max(accuracies))]
print("\nBest k:", best_k)

# Grid Search
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

pivot = grid_df.pivot_table(
    index="param_n_neighbors",
    columns="param_weights",
    values="mean_test_score"
)

plt.figure(figsize=(6, 4))
sns.heatmap(pivot, annot=True, cmap="viridis")
plt.title("GridSearchCV Heatmap")
plt.show()

# Random Search
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

plt.figure(figsize=(6, 4))
plt.hist(scores, bins=10)
plt.title("RandomizedSearchCV Score Distribution")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()

# KDTree vs BallTree
print("\nKDTree vs BallTree")

for algo in ["kd_tree", "ball_tree"]:

    model = KNeighborsClassifier(
        n_neighbors=best_k,
        algorithm=algo
    )

    train_time, pred_time = measure_time(
        model,
        X_train,
        y_train,
        X_test
    )

    pred = model.predict(X_test)

    print("\nAlgorithm:", algo)
    print("Accuracy :", accuracy_score(y_test, pred))
    print("Training :", train_time)
    print("Prediction:", pred_time)

# Cross Validation
print("\n5-Fold Cross Validation")

cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_scores[name] = scores

    print("\n", name)
    print(scores)
    print("Average:", scores.mean())

plt.figure(figsize=(7, 4))
plt.bar(
    cv_scores.keys(),
    [score.mean() for score in cv_scores.values()]
)
plt.ylabel("Accuracy")
plt.title("Cross Validation Accuracy")
plt.xticks(rotation=15)
plt.show()

# Timing
time_df = pd.DataFrame(
    times,
    columns=["Model", "Training", "Prediction"]
)

print("\nTraining / Prediction Time")
print(time_df)

plt.figure(figsize=(6, 4))
plt.bar(time_df["Model"], time_df["Training"])
plt.title("Training Time")
plt.ylabel("Seconds")
plt.xticks(rotation=15)
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(time_df["Model"], time_df["Prediction"])
plt.title("Prediction Time")
plt.ylabel("Seconds")
plt.xticks(rotation=15)
plt.show()
