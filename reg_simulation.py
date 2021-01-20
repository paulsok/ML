# import simulation of the regression data
from sklearn.datasets import make_regression

features, target, coeff = make_regression(n_samples=100, n_features=3,
                                          n_informative=3, n_targets=1,
                                          noise=0.2, coef=True, random_state=1)
