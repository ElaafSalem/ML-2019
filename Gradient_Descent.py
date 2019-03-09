import math 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('training.csv', names=['x','y'])
X = pd.DataFrame(df, columns=['x'])
Y = pd.DataFrame(df, columns=['y'])

eta = 0.001 # learning rate
i = 1000 # iteration number

X_bar = np.c_[np.ones((len(X), 1)), X] # x vectors 

theta = [[2],[3]] # coefficient vectors

for iteration in range(0, i):
    theta = theta - ((1.0/m * X_bar.T.dot(X_bar.dot(theta) - Y) * eta)) # gradient descent

X_predict = np.array([[0],[5]])  # (min, max)
X_bar_perdict = np.c_[np.ones((2,1)), X_predict] 
Y_predict = X_bar_perdict.dot(theta)
plt.plot(X, Y, 'b.')
plt.plot(X_predict, Y_predict, 'r')

df = pd.read_csv('testing.csv', names=['x','y'])
X_new = pd.DataFrame(df, columns=['x'])
Y_new = pd.DataFrame(df, columns=['y'])

plt.plot(X_new, Y_new, 'b.')
plt.plot(X_predict, Y_predict, 'r')

def rmse_metric(actual, predicted):
    sum_error = 0.0
    prediction_error = np.asarray(predicted) - np.asarray(actual)
    sum_error += (prediction_error ** 2)
    sum = 0
    for i in sum_error:
        sum+=i
    mean_error = sum / float(len(actual))
    return math.sqrt(mean_error)

Y_test_p = X_new*theta[1] + theta[0]
print(rmse_metric(Y_new, Y_test_p))