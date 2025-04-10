import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#import required libraries
df=pd.read_csv(r"C:\Users\thiru\OneDrive\Desktop\cardio_train.csv", delimiter=";")    
#Data Preprocessing
print("Checking for missing values in the dataset:")
missing_values = df.isnull().sum()
print(missing_values)
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()
numerical_columns = df.select_dtypes(include=[np.number]).columns
for col in numerical_columns:
    df[col].fillna(df[col].mean(), inplace=True)
categorical_columns = df.select_dtypes(include=[object]).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)
print("\nMissing values after filling:")
missing_values_after = df.isnull().sum()
print(missing_values_after)
#Data Analysis & Visualizations
print("\n Performing Exploratory Data Analysis & Visualizations...\n")
print(" First 5 Rows of Dataset:")
print(df.head())
print("\n Dataset Info:")
print(df.info())
print("\n Checking for Missing Values:")
print(df.isnull().sum())
print("\n Summary Statistics:")
print(df.describe())
print("\n Columns that can be used for analysis:")
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
print(numeric_columns)
selected_columns = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "cardio"]
sns.pairplot(df[selected_columns], hue="cardio", diag_kind="kde", palette="husl")
plt.show()
#Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True)   
plt.title("Correlation Matrix Heatmap")
plt.show()
# Split dataset into features and target variable
X = df.drop(columns=['cardio'])  # Features
y = df['cardio']  # Target variable
# Split into training and test sets (without random_state)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Initialize models
models = {"Support Vector Machine": SVC(),"K-Nearest Neighbors": KNeighborsClassifier(),"Decision Tree": DecisionTreeClassifier(),"Logistic Regression": LogisticRegression(),"Random Forest": RandomForestClassifier()}
# Train and evaluate each model using .score()
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)  # Using .score()
    results[name] = accuracy

# Print accuracy results
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")
# Choose the best model
best_model = max(results, key=results.get)
print(f"\nBest Model: {best_model} with accuracy {results[best_model]:.4f}")
print(results)
