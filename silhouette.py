from __future__ import absolute_import, division, print_function, unicode_literals

import pandas as pd
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

# For each k value, we will initialise k-means and use the inertia attribute to identify the sum of squared distances
# of samples to the nearest cluster centre.
silhouette_coefficients = []
K = range(2,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(train_transformed)
    score = silhouette_score(train_transformed, km.labels_)
    silhouette_coefficients.append(score)

# Silhouette graph
plt.plot(K, silhouette_coefficients, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Coefficient For Optimal k')
plt.show()
