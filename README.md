# House-Price-Predictor
Project to predict house prices 
# 🏠 House Price Predictor (Linear Regression)

A simple Machine Learning project to **predict house prices** using **Linear Regression** with the **Boston Housing dataset**.

> This project is mainly for learning and practice: EDA, data visualization, feature correlation, train/test split, and basic regression evaluation (MSE/RMSE).

---

## 📌 Project Overview

This notebook includes:

- Load dataset (Boston Housing)
- Basic data inspection (`shape`, `describe`, missing values)
- Exploratory Data Analysis (EDA)
  - Pairplot visualization
  - Distribution plot of target value
  - Skewness & Kurtosis analysis
  - Correlation heatmap
- Apply **log transformation** on target (`Sale Price`)
- Train **Linear Regression** model
- Evaluate model using:
  - **MSE (Mean Squared Error)**
  - **RMSE (Root Mean Squared Error)**

---

## 📊 Dataset

This project uses the **Boston Housing dataset**, with these features:

- `CRIM` : per capita crime rate by town  
- `ZN` : proportion of residential land zoned for lots over 25,000 sq.ft.  
- `INDUS` : proportion of non-retail business acres per town  
- `CHAS` : Charles River dummy variable (1 if tract bounds river; 0 otherwise)  
- `NOX` : nitric oxides concentration  
- `RM` : average number of rooms per dwelling  
- `AGE` : proportion of owner-occupied units built prior to 1940  
- `DIS` : weighted distances to five Boston employment centres  
- `RAD` : index of accessibility to radial highways  
- `TAX` : full-value property-tax rate  
- `PTRATIO` : pupil-teacher ratio  
- `B` : (historical variable from dataset)  
- `LSTAT` : % lower status of the population  

🎯 Target column:
- `Sale Price` (house price)

⚠️ Note: This dataset is widely used for education, but it also contains historical issues and is not recommended for real-world deployment.

---

## 🧠 Model Used

✅ **Linear Regression** from `sklearn.linear_model`

The training process:

1. Split data into train/test (80/20)
2. Train model with `fit(X_train, y_train)`
3. Predict with `predict(X_test)`
4. Evaluate using MSE and RMSE

---

## 🛠️ Tech Stack

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- SciPy
- Scikit-learn

---
