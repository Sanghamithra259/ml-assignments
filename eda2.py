import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

sns.set_style("whitegrid")


def perform_eda(file_path):

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

    # Replace null-like values
    null_values = [
        "No Info", "Noinfo", "No Data", "None",
        "NULL", "Null", "null",
        "N/A", "NA",
        "NaN", "nan",
        "Unknown", "unknown",
        "?", "", " ", "-", "--"
    ]

    df.replace(null_values, np.nan, inplace=True)

    # Remove leading/trailing spaces
    object_cols = df.select_dtypes(include="object").columns

    for col in object_cols:
        df[col] = df[col].str.strip()

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

    print("\nCategorical Statistics")
    print(df.describe(include="object"))

    # Separate columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    print("\nNumeric Columns")
    print(numeric_cols)

    print("\nCategorical Columns")
    print(categorical_cols)

    # Remove outliers
    print("\nRemoving Outliers (IQR Method)")

    original_shape = df.shape

    for col in numeric_cols:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()

        print(f"{col}: {outliers} outliers removed")

        df = df[
            (df[col] >= lower) &
            (df[col] <= upper)
        ]

    print("\nShape Before Outlier Removal:", original_shape)
    print("Shape After Outlier Removal:", df.shape)

    # Numerical graphs
    if len(numeric_cols) > 0:

        cols = 2
        rows = math.ceil(len(numeric_cols) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))

        if rows == 1:
            axes = np.array(axes).reshape(1, -1)

        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):

            sns.histplot(
                df[col],
                bins=30,
                kde=True,
                ax=axes[i]
            )

            axes[i].set_title(f"{col} Distribution")

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

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

        plt.tight_layout()
        plt.show()

    # Correlation matrix
    if len(numeric_cols) > 1:

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            df[numeric_cols].corr(),
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Matrix")
        plt.show()

    return df


clean_df = perform_eda("diabetes_prediction_dataset.csv")
