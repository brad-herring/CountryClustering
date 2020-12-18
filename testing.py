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
    n_clusters=4,
    n_init=10,
    max_iter=300,
    random_state=42
)

# Perform the test
kmeans.fit(train_transformed)

# Convert kmeans.labels_ numpy array to Pandas dataframe
cluster_df = pd.DataFrame(kmeans.labels_, columns=['cluster'])

# Check the code
print(train_target[0])
print(kmeans.labels_[0])
print(train_target[159])
print(kmeans.labels_[159])
print(str(train_target[0]) + ' - Cluster: ' + str(cluster_df['cluster'][0]))
print(str(train_target[159]) + ' - Cluster: ' + str(cluster_df['cluster'][159]))


for item in cluster_df['cluster']:
    index = cluster_df[cluster_df['cluster'] == 2].index[0]
    print(train_target[index])
