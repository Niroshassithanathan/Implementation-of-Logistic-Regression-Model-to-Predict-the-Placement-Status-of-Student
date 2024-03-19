# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Step 1 :
Import the standard libraries such as pandas module to read the corresponding csv file.

### Step 2 :
Upload the dataset values and check for any null or duplicated values using .isnull() and .duplicated() function respectively.

### Step 3 :
Import LabelEncoder and encode the corresponding dataset values.

### Step 4 :
Import LogisticRegression from sklearn and apply the model on the dataset using train and test values of x and y.

### Step 5 :
Predict the values of array using the variable y_pred.

### Step 6 :
Calculate the accuracy, confusion and the classification report by importing the required modules such as accuracy_score, confusion_matrix and the classification_report from sklearn.metrics module.

### Step 7 :
Apply new unknown values and print all the acqirred values for accuracy, confusion and the classification report.

### Step 8:
End the program.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: NIROSHA S
RegisterNumber: 212222230097 
*/
```
```
import pandas as pd

data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x = data1.iloc[:,:-1]
x
y = data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
HEAD OF THE DATA :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/7d867b1d-1942-4470-9f16-04d988bfa01f)

COPY HEAD OF THE DATA:

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/ee207f05-2131-4252-a6e2-9873c0e8fc31)

NULL AND SUM :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/ed6a419e-b440-472d-9642-d727d9b1aa40)


DUPLICATED :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/db3edf6f-b7fb-4fb7-baec-3876067018fe)

X VALUE:

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/bfdff9f8-af25-422d-bad8-f4457e8349f0)

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/2d6918db-40bc-449c-996c-0dee1d59b15c)

Y VALUE :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/e8c8654f-a62d-4189-b6a1-071b36e38364)


PREDICTED VALUES :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/98ceef5f-abf1-4c1a-a7a8-2f3c3eab8df4)


ACCURACY :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/4aab5c65-7fd9-400d-b1d7-4abf36c1ff94)

CONFUSION MATRIX :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/ba0a17b6-89fb-4526-b20c-e8b1a315f2da)


CLASSIFICATION REPORT :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/b0c48ca8-2cbf-4d5f-bd44-cf68771d804e)


Prediction of LR :

![image](https://github.com/Niroshassithanathan/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/121418437/1f4c078d-e713-4589-bf0c-7db945821f3e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
