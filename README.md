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

## Experimentation & Model Improvement

This project was built through multiple iterations rather than a single implementation. Instead of stopping after training an XGBoost classifier, I experimented with different techniques to improve both the model and my understanding of machine learning.

### Feature Engineering

I created and tested several engineered features, including:

* Balance Per Product
* Salary to Balance Ratio
* Senior Citizen Indicator
* Loyal Customer Indicator
* Credit Income Ratio

After evaluating different combinations, I found that these additional features did not improve the model's performance. The original feature set produced better results, so it was retained for the final model. This experiment reinforced the importance of validating ideas with data rather than assumptions.

### Cross Validation

To obtain a more reliable estimate of model performance, I evaluated the model using **5-Fold Cross Validation** instead of relying solely on a single train-test split. This helped measure how well the model generalizes across different data splits.

### Hyperparameter Tuning

I used **GridSearchCV** to automatically search for the best XGBoost hyperparameters instead of selecting them manually. The search identified the optimal combination of parameters that achieved the best cross-validation performance.

### Feature Importance

Using XGBoost's built-in feature importance analysis, I identified which features contributed the most to the model's decisions. The analysis showed that variables such as **Age**, **Balance**, and **Credit Score** played significant roles, while features like **Has Credit Card** had comparatively little impact.

### Explainable AI with SHAP

To better understand the model's predictions, I implemented **SHAP (SHapley Additive exPlanations)**. SHAP provides visual explanations showing how each feature influences the prediction for both the overall model and individual customers. This made the model more interpretable rather than functioning as a black box.

### Key Learning

This project taught me that machine learning is not simply about training a model. It involves experimentation, evaluation, optimization, and explainability. Building and improving the model through multiple iterations provided a much deeper understanding of the complete machine learning workflow.


## Author

Adithya Prasad
