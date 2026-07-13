import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set graph style
sns.set_style("whitegrid")

def perform_eda(file_path):

    # Load the dataset
    df = pd.read_csv(file_path)

    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Show first 5 rows
    print("\nFirst 5 Rows")
    print(df.head())

    # Show last 5 rows
    print("\nLast 5 Rows")
    print(df.tail())

    # Show dataset size
    print("\nShape")
    print(df.shape)

    # Show dataset information
    print("\nDataset Info")
    df.info()

    # Show column names
    print("\nColumn Names")
    print(df.columns.tolist())

    # Show data types
    print("\nData Types")
    print(df.dtypes)

    # Check missing values
    print("\nMissing Values")
    print(df.isnull().sum())

    # Check duplicate rows
    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    # Count unique values
    print("\nUnique Values")
    print(df.nunique())

    # Show statistics
    print("\nStatistical Summary")
    print(df.describe(include="all"))

    # Remove duplicate rows
    df = df.drop_duplicates()

    print("\nShape After Removing Duplicates")
    print(df.shape)

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=np.number).columns

    # Get categorical columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    print("\nNumerical Columns")
    print(list(numerical_cols))

    print("\nCategorical Columns")
    print(list(categorical_cols))

    # Plot histograms
    if len(numerical_cols) > 0:
        df[numerical_cols].hist(figsize=(15, 10), bins=20)
        plt.suptitle("Histograms")
        plt.tight_layout()
        plt.show()

    # Plot boxplots
    for col in numerical_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot - {col}")
        plt.show()

    # Plot countplots
    for col in categorical_cols:
        plt.figure(figsize=(7, 4))
        sns.countplot(data=df, x=col)
        plt.title(f"Count Plot - {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Plot correlation heatmap
    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[numerical_cols].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    return df


# Call the function
df = perform_eda("your_dataset.csv")
