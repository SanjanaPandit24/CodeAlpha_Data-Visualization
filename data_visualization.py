import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- FIX FOR "No such file or directory" ERROR ---
# This finds the csv file automatically wherever your python file is
try:
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, "titanic.csv")
except NameError:
    file_path = "titanic.csv"

# Load the dataset
df = pd.read_csv(file_path)
print(f"Dataset loaded successfully! Shape: {df.shape}")
print(df.head())

# --- DATA CLEANING ---
# Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# Fill missing Embarked with mode (most common value)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Set plot style
sns.set_style("whitegrid")

# --- 8 KEY VISUALIZATIONS ---

# 1. Overall Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='pastel')
plt.title('1. Overall Survival Count')
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.savefig('1_survival_count.png')
plt.show()

# 2. Gender vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='viridis')
plt.title('2. Survival by Gender')
plt.savefig('2_gender_vs_survival.png')
plt.show()

# 3. Passenger Class vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('3. Survival by Passenger Class')
plt.savefig('3_class_vs_survival.png')
plt.show()

# 4. Age Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Age'], bins=30, kde=True, color='orange')
plt.title('4. Age Distribution of Passengers')
plt.savefig('4_age_distribution.png')
plt.show()

# 5. Fare vs Survival
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Fare', data=df, palette='coolwarm')
plt.title('5. Fare vs Survival')
plt.savefig('5_fare_vs_survival.png')
plt.show()

# 6. Embarked Port vs Survival
plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', hue='Survived', data=df, palette='muted')
plt.title('6. Survival by Embarked Port')
plt.savefig('6_embarked_vs_survival.png')
plt.show()

# 7. Age vs Survival
plt.figure(figsize=(8,4))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='rocket')
plt.title('7. Age vs Survival')
plt.savefig('7_age_vs_survival.png')
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10,6))
numeric_df = df.select_dtypes(include=['number'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('8. Correlation Heatmap of Numerical Features')
plt.savefig('8_correlation_heatmap.png')
plt.show()

print("All 8 visualizations created successfully!")