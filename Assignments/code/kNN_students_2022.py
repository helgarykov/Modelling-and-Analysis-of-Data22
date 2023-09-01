import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt


def data_reading(filename):
    train_data = pd.read_csv (filename, delimiter=",")
    test_data = pd.read_csv ("heart_simplified_test.csv", delimiter=",")
    data_numerical = train_data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR']].values
    data_combined = train_data[['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Sex', 'ChestPainType']].values
    data_labels = train_data[['HeartDisease']].values


    print("Number of samples: %i" % data_numerical.shape[0])
    print("Number of numerical features: %i" % data_numerical.shape[1])
    print("Number of combined features: %i" % data_combined.shape[1])

    return data_numerical, data_combined, data_labels


# Read the train/validation/test data from the corresp. csv-files
train_data_numerical, train_data_combined, train_data_labels = data_reading("heart_simplified_train.csv")
validation_data_numerical, validation_data_combined, validation_data_labels = data_reading("heart_simplified_validation.csv")
test_data_numerical, test_data_combined, test_data_labels = data_reading("heart_simplified_test.csv")


class NearestNeighborRegressor:
    
    def __init__(self, n_neighbors = 3, weights = [1,1], distance_metric = 'numerical'):
        """ 
       Initializes the model.

        Parameters
        ----------
        n_neighbors : The number of nearest neigbhors (default 1)
        weights : Weighting factors for numerical and categorical features
        distance_metric : The distance metric to use for predictions
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distance_metric = distance_metric

    
    def fit(self, X, t, type):
        """
        Fits the nearest neighbor regression model.
        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
        t : Array of labels [n_samples]
        type: feature types'numerical' or 'combined' 
     
        """ 
       
        # Reshape both arrays to make sure that we deal with 
        # N-dimensional Numpy arrays
        self.X = np.array(X).reshape((len(X), -1))
        self.t = np.array(t).reshape((-1, 1))

        # Create lists of strings for numerical and categorical features
        self.features = type
        self.numerical_features = [i for i in range(X.shape[1]) if self.features[i] == "numerical"]
        self.categorical_features = [i for i in range(X.shape[1]) if self.features[i] != "numerical"]
    

    def predict(self, X):
        """
        Computes predictions for a new set of points.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]

        Returns
        -------
        predictions : array-like of length n_samples
        """ 
        predictions = []
        for value in X:
            distances = []
            for i, point in enumerate(self.X):
                if self.distance_metric == 'mixed_distance':
                    distance = self.__mixedDistance(value, point, self.weights[0], self.weights[1])
                else:
                    distance = self.__numericalDistance(value, point)
                distances.append((distance, i))
            # sort the distances in ascending order
            distances = sorted(distances, key=lambda x: x[0])
            # make list of indices corresponding to the closest n_neighbors data points. 
            label_indices = [x[1] for x in distances[:self.n_neighbors]]
            # extract the labels of the closest data points 
            # and create a list of those labels
            labels = [self.t[i] for i in label_indices]
            predictions.append(np.mean(labels))
        return predictions
    

    def __numericalDistance(self, p, q):
        """
        Computes the Euclidean distance between 
        two points.
        """
        distance = math.sqrt(np.sum((q - p) ** 2))
        return distance


    def __mixedDistance(self, p, q, numerical_weight, categorical_weight):
        """
        Computes the distance between 
        two points via the pre-defined matrix.
        """
        distance = 0
        # distance for numerical features
        for value in self.numerical_features:       
            if isinstance(p[value], (int, float)):
                distance += numerical_weight * (p[value] - q[value]) ** 2   
        # distance for categorical features
        for value in self.categorical_features:     
            if p[value] != q[value]:
                distance += categorical_weight
        return distance
    

    def rmse(self, t, tp):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """
        n = len(t)
        error = (t - tp)**2
        rmse = math.sqrt(np.sum(error)/n)
        return rmse


    def accuracy(self, t, tp):
        """ Computes the RMSE for two
        input arrays 't' and 'tp'.
        """   
        n_correct = 0
        for i in range(len(t)):
            if np.round(t[i]) == np.round(tp[i]):
                n_correct += 1
        accuracy = n_correct/len(t)
        return accuracy

# QUESTION 4(1)

# Initialize the model as NearestNeighborRegressor 
kNN = NearestNeighborRegressor(distance_metric = 'numerical')
type = ['numerical'] * 4
# fit the kNN-model using the training data
kNN.fit(train_data_numerical, train_data_labels, type)  

# Evaluate the model performance on test data and find the accuracies on k
rmses = []
accuracies = []
for n_neighbors in range(1, 11):
    kNN.n_neighbors = n_neighbors
    predictions = kNN.predict(test_data_numerical)
    rmse = kNN.rmse(test_data_labels, predictions)
    rmses.append(rmse)
    accuracy = kNN.accuracy(test_data_labels, predictions)
    accuracies.append(accuracy)

plt.scatter(range(1, 11), accuracies)
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.show()

# Find the optimal number of neighbours and shift one to 
# the right to fit into range(1, 11)
optimal_neighbors = np.argmax(accuracies) + 1
print("Optimal k: %d" % optimal_neighbors)
print("RMSE for optimal k: %f" % rmses[optimal_neighbors - 1])
print("Accuracy for optimal k: %f" % accuracies[optimal_neighbors - 1])


# QUESTION 4(2)

# Initialize weights for categorical features to balance the combined distance
w_cat_factors = [0.01, 0.025, 0.05, 0.1]
type = ['numerical'] * 4 + ['categorical'] * 2
k = 5

for w in w_cat_factors:
    # Initialize kNN model with weight = 1 for numerical categories and 
    # w_cat_factors for categorical categories and 'mixed_distance' for
    # computing combined distances
    kNN = NearestNeighborRegressor(k,  weights = [1, w], distance_metric = 'mixed_distance')
    kNN.fit(train_data_combined, train_data_labels, type)
    predictions = kNN.predict(test_data_combined)
    rmse = kNN.rmse(test_data_labels, predictions)
    accuracy = kNN.accuracy(test_data_labels, predictions)
    print("w_cat_factor:", w, " RMSE:", rmse, " Accuracy:", accuracy)