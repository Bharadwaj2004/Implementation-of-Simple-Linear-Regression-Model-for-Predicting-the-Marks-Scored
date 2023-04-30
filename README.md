# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. mport the required libraries and read the dataframe.
2. Assign hours to X and scores to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B.venkata bharadwaj
RegisterNumber: 212222240020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

x=df.iloc[:,:-1].values
x

y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

y_test
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scrores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:

![image](https://user-images.githubusercontent.com/119389139/230030388-5cc59fa1-179f-40a1-b039-bd91ee6b04f1.png)

![image](https://user-images.githubusercontent.com/119389139/230030496-11aee79a-91f1-4641-88a3-9a63ef1742e3.png)

![image](https://user-images.githubusercontent.com/119389139/230030599-fdde8425-5b46-425e-a4b9-45cd5d363436.png)

![image](https://user-images.githubusercontent.com/119389139/230030817-fcb96383-b463-4ac3-a5fc-ac5dfcbf5721.png)

![image](https://user-images.githubusercontent.com/119389139/230031140-bf4c46e9-dc2f-4cc4-b23c-37d6d73fcc62.png)

![image](https://user-images.githubusercontent.com/119389139/230033037-d05b409c-8022-48b9-b423-a116b81090b5.png)

![image](https://user-images.githubusercontent.com/119389139/230034346-339ed1aa-9572-4a0b-aca6-5fbf458cded9.png)

![image](https://user-images.githubusercontent.com/119389139/230033991-1ca0ee04-3ca5-4dde-a0bd-db2935d38122.png)

![image](https://user-images.githubusercontent.com/119389139/230033345-669e2734-ae50-4242-92f4-fa526a6fd3df.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
