# Customer Churn Predictor using XGBoost

## About

This project predicts whether a bank customer is likely to leave the bank based on customer information such as age, credit score, balance, tenure, salary, and account details. The goal of this project was not only to build a machine learning model but also to understand the complete workflow involved in solving a real-world classification problem.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost

---

## Dataset

The dataset contains customer information including:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Has Credit Card
* Active Member Status
* Estimated Salary

Target Variable:

* **0 → Customer Stays**
* **1 → Customer Churns**

---

## Workflow

```
Load Dataset
        ↓
Data Preprocessing
        ↓
Remove Unnecessary Columns
        ↓
One-Hot Encoding
        ↓
Train-Test Split
        ↓
Train XGBoost Classifier
        ↓
Evaluate Accuracy
        ↓
Interactive Prediction
```

---

## Experimentation

This project involved several experiments rather than simply training a model.

I explored feature engineering by creating additional features such as:

* Balance Per Product
* Salary to Balance Ratio
* Senior Citizen Indicator
* Loyal Customer Indicator
* Credit Income Ratio

After testing different combinations, I found that these engineered features did not improve the model's performance. The original feature set achieved the best accuracy, so it was retained for the final implementation.

This experience taught me that feature engineering should always be validated through experimentation and that adding more features does not necessarily produce a better model.

---

## Features

* Customer churn prediction
* Data preprocessing
* One-Hot Encoding for categorical variables
* XGBoost classification model
* Interactive prediction using user input
* Accuracy evaluation

---

## What I Learned

Through this project, I learned:

* The difference between classification and regression problems
* Why One-Hot Encoding is required for categorical data
* How XGBoost works for structured datasets
* The importance of experimentation in machine learning
* That feature engineering is hypothesis-driven and should always be evaluated using model performance

---

## Future Improvements

* Hyperparameter tuning
* Cross Validation
* Feature Importance Analysis
* Model comparison with other algorithms
* Deployment as a web application

---

## Author

**Adithya Prasad**
