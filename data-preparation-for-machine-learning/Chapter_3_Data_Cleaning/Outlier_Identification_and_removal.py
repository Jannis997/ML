# For data with a gaussian distribution
# Remove datapoints that are 2 to 4 standart deviations away from the mean
# 2 for small and 4 for large datasets
from numpy import mean
from numpy import std
# calculate summary statistics
data_mean, data_std = mean(data), std(data)
# define outliers
cut_off = data_std * 3
lower, upper = data_mean - cut_off, data_mean + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]

# outlier detection with the IQR-method
from numpy import percentile
# calculate interquartile range
q25, q75 = percentile(data, 25), percentile(data, 75)
iqr = q75 - q25
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
# remove outliers
outliers_removed = [x for x in data if x >= lower and x <= upper]

# outlier detection with LOF
# givesscores to each instance dependend on nearest neighborhood
# identify outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)
# select all rows that are not outliers
mask = yhat != -1
X_train, y_train = X_train[mask, :], y_train[mask]

# Other outlier detection algorithms:
# IsolationForest
