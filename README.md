# 🚢 Titanic Survival Prediction - Machine Learning Project

## 📖 Description

This project predicts passenger survival from the Titanic disaster using machine learning. The goal is to analyze the characteristics of passengers and determine which groups were more likely to survive.

The Titanic dataset is a well-known benchmark in data science, providing rich opportunities for feature engineering, data cleaning, and model evaluation.

---

## 🎯 Objective

- Understand and preprocess the Titanic dataset  
- Engineer new meaningful features  
- Build and train a machine learning pipeline  
- Evaluate model performance using accuracy and confusion matrix  
- Visualize results for interpretation  

---

## 🧰 Technologies Used

- Python 3  
- NumPy, Pandas  
- Scikit-learn  
- Matplotlib  
- Jupyter Notebook  

---

## 📁 Dataset Overview

| Column       | Description                                  |
|--------------|----------------------------------------------|
| PassengerId  | Unique ID for each passenger                 |
| Survived     | Target variable (0 = No, 1 = Yes)            |
| Pclass       | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)      |
| Name         | Passenger's full name                        |
| Sex          | Gender                                       |
| Age          | Age in years                                 |
| SibSp        | # of siblings/spouses aboard                 |
| Parch        | # of parents/children aboard                 |
| Ticket       | Ticket number                                |
| Fare         | Fare paid for the ticket                     |
| Cabin        | Cabin number                                 |
| Embarked     | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## 🧪 Feature Engineering

New features were created to improve model performance:

| Feature       | Type       | Description |
|---------------|------------|-------------|
| `FamilySize`  | Numerical  | Total family size: `SibSp + Parch + 1` |
| `IsAlone`     | Numerical  | 1 if passenger has no family aboard, else 0 |
| `Title`       | Nominal    | Extracted from the `Name` column (e.g., Mr, Miss, etc.) |

These features help capture social status, dependency, and travel context.

---

## 🔄 Preprocessing Pipeline

Using Scikit-learn's `Pipeline` and `ColumnTransformer`, we:

- Imputed missing values (median for numerical, most frequent for categorical)
- Standardized numeric features using `StandardScaler`
- Applied PCA for dimensionality reduction
- Encoded nominal and ordinal features
- Performed feature selection using `SelectKBest` and `RFE`

---

## 🏗️ Model Pipeline

A complete pipeline was built with:

- Preprocessing steps  
- Recursive Feature Elimination (RFE)  
- Logistic Regression as the classifier

