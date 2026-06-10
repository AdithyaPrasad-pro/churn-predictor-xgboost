import pandas as pd
from sklearn.model_selection import cross_val_score

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_churn.csv")

x=df.drop(["RowNumber","Exited","CustomerId","Surname","Exited"],axis = 1)

y=df["Exited"]

x = pd.get_dummies(x,columns=["Geography","Gender"],drop_first=True)

model = XGBClassifier(random_state=42)

scores = cross_val_score(model,x,y,cv=5,scoring="accuracy")


model.fit(x,y)

print("Cross Validation Scores:")
print(scores)
print()
print(f"Average Accuracy: {scores.mean()*100:.2f}%")


creditscore =int(input("Credit Score: "))
geography = input("Geography (France/Germany/Spain): ").strip().lower()
gender =input("Gender (Male/Female): ").strip().lower()
age =int(input("Age: "))
tenure= int(input("Tenure: "))
balance= float(input("Balance: "))
num_products= int(input("Number of Products: "))
has_card =int(input("Has Credit Card (1=Yes,0=No): "))
active =int(input("Is Active Member (1=Yes,0=No): "))
salary =float(input("Estimated Salary: "))


geo_germany= 1 if geography == "germany" else 0
geo_spain =1 if geography == "spain" else 0
gender_male= 1 if gender == "male" else 0


user_data = pd.DataFrame([{
    "CreditScore": creditscore,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary,
    "Geography_Germany": geo_germany,
    "Geography_Spain": geo_spain,
    "Gender_Male": gender_male
}])

prediction = model.predict(user_data)

if prediction[0] == 1:
    print("\nPrediction: Customer is likely to Churn")

else:print("\nPrediction: Customer is likely to Stay")

