# Iterative Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
# define imputer
imputer = IterativeImputer(estimator=BayesianRidge(), n_nearest_features=None,
imputation_order='ascending')
# fit on the dataset
imputer.fit(X)
# transform the dataset
Xtrans = imputer.transform(X)
