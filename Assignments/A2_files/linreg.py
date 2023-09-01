import numpy as np

from housing_1 import t_train

# NOTE: This template makes use of Python classes. If 
# you are not yet familiar with this concept, you can 
# find a short introduction here: 
# http://introtopython.org/classes.html

class LinearRegression():
    """
    Linear regression implementation.
    """

    def __init__(self):
        
        pass
            
    def fit(self, X, t):
        """
        Fits the linear regression model.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of shape [n_samples, 1]
        """       

        X=np.array(X).reshape((len(X), -1))
        t= np.array(t).reshape((len(t),1)) 

        # prepend a column of ones
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)       #concatenate two NumPy arrays column-wise

        # compute weights  (matrix inverse)
        self.w = np.linalg.solve((np.dot(X.T, X)),np.dot(X.T, t))
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
