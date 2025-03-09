# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for the marks by using the representation of the graph.
6. Hence we obtained the linear regression for the given dataset.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VENKATA BHARADWAJ.B
RegisterNumber:  212222240020
*/
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ml CSV.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
x_train
y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='orange')
lr.coef_
lr.intercept_
```

## Output:

## 1. Dataset:

![2 1](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/39044f33-3de2-4dba-a406-4fc8966b7dc2)

## 2.  Graph of plotted data:

![2 2](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/31e506d3-4a00-49ee-8ae7-80540735b946)

## 3.  Performing Linear Regression:

![2 3](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/1d7ffebc-7c31-4ccb-945b-de920d2b132f)

## 4.  Trained data:

![2 4](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/ff2f5e4d-e483-471c-943f-cd8caecf318d)

## 5.  Predicting the line of Regression:

![2 5](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/63b7de8b-0137-45b1-9c4c-90b3e3dab59c)

## 6.  Coefficient and Intercept values:

![2 6](https://github.com/SaiPraneeth04/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119390353/57d33c39-9c94-4730-9895-665674a76709)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
