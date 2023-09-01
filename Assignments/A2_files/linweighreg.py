import numpy as np
import matplotlib.pyplot as plt
from housing_1 import X_train, t_train, X_test, t_test
import linreg 

class WeightedLinearRegression():
    """
    Non-Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def fit(self, X, t):
        """
        Fits the non-linear regression model.

        Parameters
        ----------
        X : Data matrix array of shape [n_features, 1]
        t : Target vector array of shape [n_features, 1]
        """       
        # reshape both arrays to make sure that we deal with 
        # N-dimensional Numpy arrays
        X=np.array(X).reshape((len(X), -1))
        t= np.array(t).reshape((len(t), 1)) 

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)      

        # generate weight-points for the weighted average loss
        weight = (t**2)
        # create a diagonal matrix that contains weights on the diagonal
        A =np.diagflat(weight)
    
        
        # compute optimal coefficients for the weighted average loss model
        self.w = np.linalg.solve(np.dot(X.T, np.dot(A, X)),np.dot(X.T, np.dot(A, t)))
        return self

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]

        Returns
        -------
        predictions : Array of shape [n_samples, 1]
        """                     
        X=np.array(X).reshape((len(X), -1))
       
        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)  

        predicted_feature = np.dot(X,self.w)
        return predicted_feature 

# (Assignment 1) fit linear regression model using all features
model_all = linreg.LinearRegression()
model_all.fit(X_train[:,: -1], t_train)

# evaluation of results
model_predicton_all = model_all.predict(X_train[:,:-1])
plt.scatter (t_train, model_predicton_all)
plt.show()

# (1b) fit  weighted linear regression model using all features
model_all = WeightedLinearRegression()
model_all.fit(X_train[:,: -1], t_train)

# evaluation of weighted results
model_predict_all = model_all.predict(X_test[:,:-1])
plt.scatter (t_test, model_predict_all)
plt.show()

