import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Titanic Dashboard", layout="wide")
st.title("🚢 Titanic Data Visualization - CodeAlpha Task 3")
st.markdown("---")

# Load data
df = pd.read_csv('titanic.csv')
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

st.success(f"Dataset Loaded Successfully! Shape: {df.shape}")
st.dataframe(df.head())

# 8 Graphs Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Overall Survival Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Survived', data=df, palette='pastel', ax=ax)
    st.pyplot(fig)

    st.subheader("3. Passenger Class vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2', ax=ax)
    st.pyplot(fig)

    st.subheader("5. Fare vs Survival")
    fig, ax = plt.subplots()
    sns.boxplot(x='Survived', y='Fare', data=df, palette='coolwarm', ax=ax)
    st.pyplot(fig)

    st.subheader("7. Age vs Survival")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='Survived', bins=30, kde=True, palette='rocket', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("2. Gender vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x='Sex', hue='Survived', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

    st.subheader("4. Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Age'], bins=30, kde=True, color='orange', ax=ax)
    st.pyplot(fig)

    st.subheader("6. Embarked Port vs Survival")
    fig, ax = plt.subplots()
    sns.countplot(x='Embarked', hue='Survived', data=df, palette='muted', ax=ax)
    st.pyplot(fig)

    st.subheader("8. Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
    st.pyplot(fig)

st.markdown("---")
st.write("Made by You for CodeAlpha Internship")