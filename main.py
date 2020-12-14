from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from numpy import nan
from matplotlib import pyplot as plt

# Create training and testing datasets
train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\Datasets\Country-data.csv', header=0)

test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\Datasets\Country-data.csv', header=0)

plt.scatter(train['country'], train['gdpp'], s=0.5)
plt.show()

# Use elbow method to determine number of k-means clusters
# Use sklearn clustering tools to carry out the study
