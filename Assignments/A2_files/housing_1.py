import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib
import plotly


# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))  #returns a column-vector
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i" % X_train.shape[1])

# (a) compute mean of prices on training set
test_mean = np.mean(t_test)
print("Mean of the house prices: %i\n" % test_mean)

# (b) RMSE function
def rmse(t, tp):
    mean_square_error = np.square(np.subtract(t,tp)).mean() 
    rsme = math.sqrt(mean_square_error)
    return rsme

tp = np.linspace(test_mean, test_mean, len(t_test))  # returns an array of train_mean vals
test_rmse = rmse(t_test, tp)
print("RMSE: %.3f\n" % test_rmse)

# (c) visualization of results: the 2D scatter plot
plt.scatter (np.arange(0, len(t_test),1), tp) 
plt.show()

# numbers = np.arange(1, 11, 2)
# print(numbers)
# output: [1 3 5 7 9]


