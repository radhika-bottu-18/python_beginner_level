
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r'D:\DataScienceAndAICourse\April\11th, 14th- NAIVE BAYES\Social_Network_Ads.csv')
x= dataset.iloc[:,[2,3]].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import Normalizer
sc= Normalizer()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import MultinomialNB 
classifier = MultinomialNB()
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusin matrix : \n',cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test,y_pred)
print('accuracy score: ',ac)

bias =classifier.score(X_train,y_train)
print(bias)