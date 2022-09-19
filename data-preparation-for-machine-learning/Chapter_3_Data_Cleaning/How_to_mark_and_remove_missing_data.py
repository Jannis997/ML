# To find out in which coulumn are how many missing values (works only for nan):
data.info()
# or
data.describe()

# Sometimes missing data is not represented with nan but with 0 or -1
# In this case first replace the values in the corresponding columns
# replace '0' values with 'nan'
dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, nan)
# followed by
print(dataset.isnull().sum())
# or
data.info()
# or
data.describe()

# drop rows with missing values
dataset.dropna(inplace=True)
