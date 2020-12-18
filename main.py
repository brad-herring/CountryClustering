from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from kneed import KneeLocator
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Create training and testing datasets
train = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\Datasets\Country-data.csv', header=0)

test = pd.read_csv(r'C:\Users\Admin\Desktop\Programming Applications and Projects\Datasets\Country-data.csv', header=0)

# Create training and testing targets
train_target = train.pop('country')
test_target = test.pop('country')

# Scale (standardize) the data
mms = MinMaxScaler()
mms.fit(train)
train_transformed = mms.transform(train)

# Instantiate the KMeans class
kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

# Perform the test
kmeans.fit(train_transformed)


print(train_target[0])
print(kmeans.labels_[0])
print(train_target[159])
print(kmeans.labels_[159])
print(train_target[154])
print(kmeans.labels_[154])
print(train_target[38])
print(kmeans.labels_[38])
print(train_target[46])
print(kmeans.labels_[46])
print(train_target[132])
print(kmeans.labels_[132])

# Best -> Worst: 0, 2, 1

