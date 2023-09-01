import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import matplotlib.cm as cm
import numpy.matlib
from sklearn.metrics import average_precision_score
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


# QUESTION 6(a)

# Read the data from the three csv-files
training_data = pd.read_csv("heart_simplified_train.csv")
validation_data = pd.read_csv("heart_simplified_validation.csv")
test_data = pd.read_csv("heart_simplified_test.csv")

# Convert categorical features into numerical ones
training_data = pd.get_dummies(training_data, columns=["Sex"])
training_data = pd.get_dummies(training_data, columns=["ChestPainType"])
validation_data = pd.get_dummies(validation_data, columns=["Sex"])
validation_data = pd.get_dummies(validation_data, columns=["ChestPainType"])
test_data = pd.get_dummies(test_data, columns=["Sex"])
test_data = pd.get_dummies(test_data, columns=["ChestPainType"])

# Extract feature and label values 
training_features = training_data.drop(columns=["HeartDisease"])
training_labels = training_data["HeartDisease"]
validation_features = validation_data.drop(columns=["HeartDisease"])
validation_labels = validation_data["HeartDisease"]
test_features = test_data.drop(columns=["HeartDisease"])
test_labels = test_data["HeartDisease"]


# QUESTION 6(b)

# Initialize predictor as the random forest model and fit it to train data
def randomForestsClassifier(training_features, training_labels, criterion = "gini", max_features = "auto", max_depth = None):
    predictor = RandomForestClassifier(n_estimators=100, criterion=criterion, max_features=max_features, max_depth=max_depth)
    predictor.fit(training_features, training_labels)
    return predictor

# Compute the accuracy of the predictor model
def accuracy(predictor, validation_features, validation_labels):
    correct_preds = 0
    number_of_preds = 0
    # generate an array of probabilities for validation_features 
    # belonging to each class
    

    #predict_proba() is a common method provided by many machine
    # learning libraries, such as scikit-learn, for classifiers 
    # that support probability prediction. This method is used to 
    # predict the probability of the instances belongs to each class.
    # The output of the method is an array of shape (n_samples, n_classes), 
    # where the element (i, j) represents the probability of the i-th 
    # instance belonging to the j-th class. It is not a built-in python function.
    # It is a method of a classifier which can be used after training the classifier. 
    # It requires a trained classifier, and it will give the probabilistic outputs of
    #  the classifier for the input instances.
    probabilities = predictor.predict_proba(validation_features)

    # make  a list of probabilities of correctly predicted labels
    # in the validation set 
    probalilities_of_correct_pred = []
    correct_preds_map = []
    for idex, predicted_label in enumerate(predictor.predict(validation_features)):
        number_of_preds = number_of_preds + 1
        probalilities_of_correct_pred.append(probabilities[idex,predicted_label])

        #If the two labels match, then it increments the "correct"
        # variable by 1 and appends a value of 1 to the "correct_map"
        # list. If the labels do not match, it appends a value of 0 to the "correct_map" list.
        if(predicted_label == validation_labels[idex]):
            correct_preds = correct_preds + 1
            correct_preds_map.append(1)
            continue
        correct_preds_map.append(0)
    accuracy = correct_preds / number_of_preds
    probability_mean = np.mean(probalilities_of_correct_pred)

    return accuracy, correct_preds, probability_mean

# Initilize predictor as RandomForestClassifier on train data
predictor = randomForestsClassifier(training_features, training_labels)

# Evaluate the predictor model performance on validation data
acuracy, correct_predictions, probability_mean = accuracy(predictor, validation_features, validation_labels)
print("Test accuracy: ", acuracy)

# QUESTION 6(c)