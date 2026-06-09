# Customer Churn Predictor using XGBoost

## About

This project predicts whether a customer is likely to leave a bank based on their personal and account information. I built this project to understand how XGBoost works and to learn the importance of feature engineering before training a machine learning model.

Instead of directly training the model, I first analyzed the dataset, removed unnecessary columns, handled categorical variables using One-Hot Encoding, and then trained an XGBoost classifier to make predictions.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* XGBoost

---

## Dataset Features

The model uses customer information such as:

* Credit Score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of Products
* Credit Card Status
* Active Member Status
* Estimated Salary

The target variable is:

* **Exited = 0** → Customer Stayed
* **Exited = 1** → Customer Churned

---

## Project Workflow

```text
Load Dataset
      ↓
Analyze Features
      ↓
Remove Unnecessary Columns
      ↓
One-Hot Encode Categorical Data
      ↓
Train-Test Split
      ↓
Train XGBoost Model
      ↓
Evaluate Accuracy
      ↓
Interactive Customer Prediction
```

---

## Experimentation

While building this project, I learned that machine learning is not only about choosing an algorithm. Preparing the data correctly is equally important.

Some of the experiments and decisions included:

* Removing columns like `CustomerId` and `Surname` since they do not help prediction.
* Learning why categorical variables cannot be directly used by the model.
* Applying One-Hot Encoding for `Geography` and `Gender`.
* Understanding why this is a **classification** problem and why **Accuracy Score** is the appropriate evaluation metric.

---

## Features

* Customer churn prediction
* Data preprocessing and feature engineering
* One-Hot Encoding of categorical variables
* XGBoost classification model
* Interactive prediction using user input
* Accuracy evaluation

---

## What I Learned

Through this project, I learned:

* The difference between classification and regression problems
* Why feature engineering is important
* How One-Hot Encoding works
* Why XGBoost is a powerful algorithm for structured data
* The importance of understanding the data before training a model

---

## Future Improvements

* Perform hyperparameter tuning
* Apply Cross Validation
* Compare XGBoost with Random Forest and Logistic Regression
* Build a web application for customer churn prediction
* Deploy the model online

---

## Author

**Adithya Prasad**
