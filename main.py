import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_churn.csv")

x=df.drop(["RowNumber","Exited","CustomerId","Surname","Exited"],axis = 1)
x["BalancePerProduct"] = (x["Balance"]/x["NumOfProducts"].replace(0,1))

x["SalaryBalanceRatio"] = (x["EstimatedSalary"]/(x["Balance"]+1))

x["SeniorCitizen"] = (x["Age"] >= 60).astype(int)

x["LoyalCustomer"] = (x["Tenure"] >= 5).astype(int)

x["CreditIncomeRatio"] = (x["CreditScore"] /(x["EstimatedSalary"] + 1))


y=df["Exited"]

x = pd.get_dummies(x,columns=["Geography","Gender"],drop_first=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


model = XGBClassifier(random_state=42)

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy= accuracy_score(y_test,y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")

print("\nEnter Customer Details")

creditscore =int(input("Credit Score: "))
age =int(input("Age: "))
tenure= int(input("Tenure: "))
balance= float(input("Balance: "))
num_products= int(input("Number of Products: "))
has_card =int(input("Has Credit Card (1=Yes,0=No): "))
active =int(input("Is Active Member (1=Yes,0=No): "))
salary =float(input("Estimated Salary: "))

geography = input("Geography (France/Germany/Spain): ").strip().lower()
gender =input("Gender (Male/Female): ").strip().lower()

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

