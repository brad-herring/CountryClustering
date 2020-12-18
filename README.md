# CountryClustering
Use of K-means clustering to find socioeconomic tiers of countries around the world

Topics covered in this project:
- Use of Scikit-Learn scaler class to standardize data
- Creating a KMeans class instance from the Scikit-Learn library
- Performing the elbow method to determine proper number of k clusters
- Using Matplotlib to represent the sum of squared distances graph (elbow method)
- Calculating the silhouette coefficient of each possible value of k
- Using Matplotlib to represent silhouette coefficient per k value
- Fitting the KMeans class to the standardized data
- Converting the kmeans.labels_ Numpy array to a Pandas DataFrame
- Creating a script for printing the countries that belong to the lowest-tier socioeconomic cluster (original objective)
- Testing the entire program twice, once for k = 3, and once for k = 4 (elbow versus silhouette discrepancy)

Possible future updates:
- Use script to return the exact value needed from the elbow method
- Evaluate the clustering performance using advanced techniques

Tutorial link(s):
https://realpython.com/k-means-clustering-python/#how-to-perform-k-means-clustering-in-python
