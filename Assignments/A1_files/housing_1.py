import matplotlib.pyplot as plt
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly


# load data
train_data = np.loadtxt("boston_train.csv", delimiter=",")
test_data = np.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))  #len(t_train)= nr of rows; 1= nr of columns
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

# Giorna: c) you should plot estimate vs t_train. 
# here you are just repeatedly plotting your mean against a linspace vector
plt.scatter (np.arange(0, len(t_train),1), tp) 
plt.show()

# import matplotlib.pyplot as plt

# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
# plt.plot(x, y)
# plt.show()
# or 
#import matplotlib.pyplot as plt
#x = [1, 2, 3, 4, 5]
#y = [2, 4, 6, 8, 10]
#plt.scatter(x, y)
#plt.show()

