# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module.
2.Read the required csv file using pandas .
3.Import LabEncoder module.
4.From sklearn import logistic regression.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.print the required values.
8.End the program.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MITHUN G
RegisterNumber: 212223080030 
*/
import pandas as pd

data = pd.read_csv("Placement_Data.csv")

print("1. Placement data")
print(data.head())


data1 = data.copy()
data1= data1.drop(["sl_no","salary"],axis=1)

print("2. Salary Data")
print(data1.head())

print("3. Checking the null() function")
print(data1.isnull().sum())

print("4. Data Duplicate")
print(data1.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

data1["gender"] = lc.fit_transform(data1["gender"])
data1["ssc_b"] = lc.fit_transform(data1["ssc_b"])
data1["hsc_b"] = lc.fit_transform(data1["hsc_b"])
data1["hsc_s"] = lc.fit_transform(data1["hsc_s"])
data1["degree_t"]=lc.fit_transform(data["degree_t"])
data1["workex"] = lc.fit_transform(data1["workex"])
data1["specialisation"] = lc.fit_transform(data1["specialisation"])
data1["status"]=lc.fit_transform(data1["status"])


print("5. Print data")
print(data1)

y = data1["status"]
print("6. Data-status")
x = data1.iloc[:,:-1]
print(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")

print("7. y_prediction array")
print(lr.fit(x_train,y_train))
y_pred = lr.predict(x_test)
print(y_pred)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("8. Accuracy")
print(accuracy)


from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print("9. Confusion array")
print(confusion)


from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)

print("10. Classification report")
print(classification_report1)

prediction = [1,67,1,91,1,1,58,2,0,55,1,58.80]
print(lr.predict([prediction])) 


prediction = [1,80,1,90,1,1,90,1,0,85,1,85]
print("11. Prediction of LR")
print(lr.predict([prediction])) 
*/
```

## Output:
![image](https://github.com/user-attachments/assets/e67cc5ed-391a-4fca-949f-e07193ac004b)
![image](https://github.com/user-attachments/assets/40d95337-a959-4d44-94d7-ae374393dfa8)
![image](https://github.com/user-attachments/assets/cdf1637a-0173-4835-bbb4-f4202db6b41b)
![image](https://github.com/user-attachments/assets/e1b2c788-7ae7-4035-9012-32e259ad5859)
![image](https://github.com/user-attachments/assets/6f148f14-6edb-4e89-bd66-77c0f80f134b)
![image](https://github.com/user-attachments/assets/8ddba92a-c7bf-47e3-b83e-29979980efb0)
![image](https://github.com/user-attachments/assets/8c5ce3c6-b643-4628-8f55-45763341937a)
![image](https://github.com/user-attachments/assets/8329a0d3-de83-48cc-b6fd-4b3fcddb648b)
![image](https://github.com/user-attachments/assets/510067a0-2a78-4221-a97a-251f6a17eb4e)
![image](https://github.com/user-attachments/assets/e8ae3938-4f8e-4542-8c08-6879ab8734bc)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
