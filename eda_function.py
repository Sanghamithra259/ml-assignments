import os
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def perform_eda(file_path, target_col=None, plots_dir="plots", show=False):
    
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

    # ------------------------------------------------------------------
    # Numerical graphs -- one combined subplot grid, saved (not popped up)
    # ------------------------------------------------------------------
    if len(numeric_cols) > 0:
        cols = 2
        rows = math.ceil(len(numeric_cols) / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 5))
        if rows == 1:
            axes = np.array(axes).reshape(1, -1)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols):
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

    # ------------------------------------------------------------------
    # Categorical graphs -- one combined subplot grid, saved (not popped up)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Correlation matrix -- FIXED: annotations turned on so the actual
    # correlation values are visible instead of just a color grid.
    # ------------------------------------------------------------------
    if len(numeric_cols) > 1:
        n = len(numeric_cols)
        # Big enough canvas + small font so annotations stay legible
        # even with a large number of numeric columns.
        fig_w = max(12, n * 0.55)
        fig_h = max(10, n * 0.5)
        annot_font_size = 6 if n > 20 else (8 if n > 12 else 10)

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        sns.heatmap(
            df[numeric_cols].corr(),
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