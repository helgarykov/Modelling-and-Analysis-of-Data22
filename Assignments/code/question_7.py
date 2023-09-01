import numpy as np
import pandas as pd
import math
import random
import matplotlib.cm as cm
import numpy.matlib


# QUESTION 7(a)

# Read the data from the csv-file and normalize the data samples
raw_data = pd.read_csv("housing.csv").values.astype(np.float64)
mean_data = raw_data.mean(0)
std_data = raw_data.std(0)
norm_data = (raw_data - mean_data)/std_data

for element in norm_data.tolist():
    print(element)