import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def perform_eda(file_path):
    #Load dataset
    df = pd.read_csv(file_path)

    print("EDA Analysis")

    #Display first few rows
    print("\nDataFrame Preview")
    print(df.head())

    #Dataset information
    print("\nDataset Info")
    df.info()

    #Column names
    print("\nColumns:")
    print(df.columns.tolist())

    #Missing values
    print("\nChecking null values")
    print(df.isnull().sum())

    #Remove "No Info" from smoking_history
    if "smoking_history" in df.columns:
        df = df[df["smoking_history"].str.lower() != "No Info"]

    print("\nNull values after filtering")
    print(df.isnull().sum())

    #Duplicate values
    print("\nChecking duplicate values")
    print(df.duplicated().sum())

    #Remove duplicates
    df = df.drop_duplicates()

    print("\nShape after removing duplicates:", df.shape)

    #Data types
    print("\nData Types")
    print(df.dtypes)

    #Unique values
    print("\nNumber of unique values")
    print(df.nunique())

    #Target distribution
    print("\nDiabetes Value Counts")
    print(df["diabetes"].value_counts())

    #graphs
    fig, axes = plt.subplots(3, 2, figsize=(20, 20))

    #diabets
    sns.countplot(x="diabetes", data=df, ax=axes[0, 0])
    axes[0, 0].set_title("Diabetes Distribution")

    #gender
    sns.countplot(x="gender", data=df, ax=axes[0, 1])
    axes[0, 1].set_title("Gender Distribution")

    # age
    sns.histplot(df["age"], bins=30, kde=True, color="skyblue", ax=axes[1, 0])
    axes[1, 0].set_title("Age Distribution")

    # BMI
    sns.histplot(df["bmi"], bins=30, kde=True, color="pink", ax=axes[1, 1])
    axes[1, 1].set_title("BMI Distribution")

    #glucose
    sns.histplot(df["blood_glucose_level"], bins=30, kde=True, color="red", ax=axes[2, 0])
    axes[2, 0].set_title("Blood Glucose Level")

    # HbA1c
    sns.histplot(df["HbA1c_level"], bins=30, kde=True, color="purple", ax=axes[2, 1])
    axes[2, 1].set_title("HbA1c Level")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


    return df

#function call
perform_eda("diabetes_prediction_dataset.csv")
