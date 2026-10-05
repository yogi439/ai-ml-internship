"""Task 1: Data Exploration and Cleaning — Titanic dataset."""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(URL)

print("First five rows:")
print(df.head())
print("\nShape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
print(df.describe(include="all"))

# Standardize column names and remove duplicate rows.
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
df = df.drop_duplicates().copy()

# Clean missing values in useful modeling columns.
if "age" in df:
    df["age"] = df["age"].fillna(df["age"].median())
if "fare" in df:
    df["fare"] = df["fare"].fillna(df["fare"].median())
if "embarked" in df:
    df["embarked"] = df["embarked"].fillna(df["embarked"].mode()[0])

print("\nCleaned missing values:")
print(df.isnull().sum())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="survived")
plt.title("Titanic Survival Distribution")
plt.tight_layout()
plt.savefig("task1_survival_distribution.png", dpi=150)
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="sex", hue="survived")
plt.title("Survival by Sex")
plt.tight_layout()
plt.savefig("task1_survival_by_sex.png", dpi=150)
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", hue="survived", bins=30, kde=True, element="step")
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("task1_age_distribution.png", dpi=150)
plt.show()

print("\nTask 1 completed successfully.")
