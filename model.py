# importing libraries
import numpy as np
import pandas as pd
import pickle                                                  

train = pd.read_csv('titanic_train.csv')

sex = pd.get_dummies(train['Sex'],drop_first=True)

train.drop(['Cabin','Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

train = pd.concat([train,sex],axis=1)

train.dropna(inplace=True)

X = train.drop('Survived',axis=1)
print(X)
y = train['Survived']
print(y)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X,y)

pickle.dump(logmodel, open('model.pkl','wb'))
