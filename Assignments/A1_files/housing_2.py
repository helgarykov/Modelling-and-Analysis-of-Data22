from gettext import npgettext
import numpy
import pandas
from housing_1 import rmse
import linreg
import matplotlib.pyplot as plt

# load data
train_data = numpy.loadtxt("boston_train.csv", delimiter=",")
test_data = numpy.loadtxt("boston_test.csv", delimiter=",")
X_train, t_train = train_data[:,:-1], train_data[:,-1]
X_test, t_test = test_data[:,:-1], test_data[:,-1]
# make sure that we have N-dimensional Numpy arrays (ndarray)
t_train = t_train.reshape((len(t_train), 1))
t_test = t_test.reshape((len(t_test), 1))
print("Number of training instances: %i" % X_train.shape[0])
print("Number of test instances: %i" % X_test.shape[0])
print("Number of features: %i\n" % X_train.shape[1])

# (b) fit linear regression using only the first feature
model_single = linreg.LinearRegression()
model_single.fit(X_train[:,0], t_train)
#optimal coefficients for the model based on the first feature
print("Model coefficients (1.feature):\n", model_single.w)

# (c) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train[:,: -1], t_train)
#optimal coefficients for the model based on all features
print("Model coefficients (all features):\n", model_all.w)                                 

# (d) evaluation of results
model_prediction_single = model_single.predict(X_train[:,0])
model_predicton_all = model_all.predict(X_train[:,:-1])

plt.scatter(t_train,model_prediction_single)
plt.show()
plt.scatter (t_train, model_predicton_all)
plt.show()

print("The value of the loss function for the first feature: %.3f\n" % rmse(t_train, model_prediction_single))
print("The value of the loss function for all features: %.3f\n" % rmse(t_train, model_predicton_all))



