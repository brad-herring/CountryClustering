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
print(train_target[159])
print(kmeans.labels_[159])
print(str(train_target[159]) + ' - Cluster: ' + str(cluster_df['cluster'][159]))
print(train_target[23])
print(kmeans.labels_[23])
print(str(train_target[23]) + ' - Cluster: ' + str(cluster_df['cluster'][23]))

print('\nCountries in need of the most aid: \n')
# Print countries in category 1 (lowest category)
for item in range(0, 166):
    if cluster_df['cluster'][item] == 1:
        print(train_target[item])
