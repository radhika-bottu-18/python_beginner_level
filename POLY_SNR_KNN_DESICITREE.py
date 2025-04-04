import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

dataset = pd.read_csv(r'D:\DataScienceAndAICourse\March-Month\emp_sal.csv')

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

y_pred = lin_reg.predict(x)
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='blue')
plt.title('Linear_regression model (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

line_reg_pred = lin_reg.predict([[6.5]])
print('Linear Regression prediction= ',line_reg_pred)

from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures()
x_poly= poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.title('Salary Prediction')
plt.show()

poly_reg_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print('Polynomial regression prediction with degree 2 is =',poly_reg_pred)

# with poly degree=3
poly_reg= PolynomialFeatures(degree=3)
x_poly= poly_reg.fit_transform(x)
poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(x,y,color='red')
plt.plot(x,lin_reg2.predict(x_poly),color='blue')
plt.title('Salary Prediction')
plt.show()

poly_reg_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print('Polynomial regression prediction with degree 3 is =',poly_reg_pred)


# SVR model prediction 
from sklearn.svm import SVR 
svr_reg = SVR(kernel='sigmoid',gamma='scale',C=10000)
svr_reg.fit(x,y)
svr_model_pred= svr_reg.predict([[6.5]])
plt.scatter(x,y,color='blue')
plt.plot(x,svr_reg.predict(x),color='red')
plt.show()
print('SVR regression prediction with degree 2 is =',svr_model_pred)

# KNN Model Prediction . 
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5,weights='distance',algorithm='ball_tree')
knn_reg.fit(x,y)
knn_model_pred = knn_reg.predict([[6.5]])
plt.scatter(x,y,color='blue')
plt.plot(x,knn_reg.predict(x),color='red')
plt.show()
print('KNN mode Prediction=',knn_model_pred)

#Decision Tree Prediction
from sklearn.tree import DecisionTreeRegressor
regressor= DecisionTreeRegressor(criterion='poisson',splitter='best',max_depth=3)
regressor.fit(x,y)
decision_tree_pred= regressor.predict([[6.5]])
print('Decision Tree regression prediction  =',svr_model_pred)
plt.scatter(x, y,color='blue')
plt.plot(x,regressor.predict(x),color='red')
plt.show()

#Random Forest Prediction
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(criterion='absolute_error',n_estimators=10,max_depth=2,random_state=0)
regressor.fit(x,y)
random_pred= regressor.predict([[6.5]])
print('Random Forest prediction = ',random_pred)

from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
lazy_reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.5,random_state=0)
models,predictions= lazy_reg.fit(x_train,x_test,y_train,y_test)

