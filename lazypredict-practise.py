import lazypredict as lp
from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np

diabetes = datasets.load_diabetes()
x,y = shuffle(diabetes.data,diabetes.target,random_state=13)

x=x.astype(np.float32)

offset = int(x.shape[0]*0.9)

x_train,y_train = x[:offset],y[:offset]
x_test,y_test = x[offset:],y[offset:]

reg = LazyRegressor(verbose=0,ignore_warnings=False,custom_metric=None)

models,predictions = reg.fit(x_train,x_test,y_train,y_test)

