import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset= pd.read_csv(r'D:\DataScienceAndAICourse\April\11th - KNN\Social_Network_Ads.csv')

x= dataset.iloc[:,[2,3]].values
y =dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
xsc_train=sc.fit_transform(x_train)
xsc_test = sc.transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(xsc_train,y_train)

y_pred = classifier.predict(xsc_test)

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)

bias = classifier.score(xsc_train,y_train)
print(bias)

variance = classifier.score(xsc_test,y_test)
print(variance)