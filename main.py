from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

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

# Best -> Worst: 0, 2, 1

# Convert kmeans.labels_ numpy array to Pandas dataframe
cluster_df = pd.DataFrame(kmeans.labels_, columns=['cluster'])

print('\nCountries in need of the most aid: \n')
# Print countries in category 1
for item in range(0, 166):
    if cluster_df['cluster'][item] == 1:
        print(train_target[item])