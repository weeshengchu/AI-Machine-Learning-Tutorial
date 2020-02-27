# 1.1 environment setup using activate tensor
# import tensorflow
# import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model

import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

# 1.2 data loading
data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())

predict = 'G3'

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])
# print(X)
# print(y)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# 1.4 training and getting the best model from 30 models
'''
best = 0
for _ in range(30):
# 1.3 implementing our training process using linear regression model
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        # 1.4 saving our best model using pickle
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

# 1.4 open the best model out of the 30 trained models
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# y = mx + c
print("Co: \n", linear.coef_)
print('Intercept: \n', linear.intercept_)

# predicting student grade with the above dataset input
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# 1.4 plotting the date to visualise
p = 'absences'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel('Final Grade')
pyplot.show()