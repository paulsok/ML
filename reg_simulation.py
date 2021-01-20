import matplotlib.pyplot as plt
# import simulation of the regression data
from sklearn.datasets import make_regression


features, target, coeff = make_regression(n_samples=100, n_features=3,
                                          n_informative=3, n_targets=1,
                                          noise=0.2, coef=True, random_state=1)

print('Features matrix\n', features[:3])
print('Targets matrix\n', target[:3])

plt.scatter(features[:, 0], features[:, 1], c=target)
plt.show()
