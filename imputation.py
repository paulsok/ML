import numpy as np
from sklearn.impute import SimpleImputer as Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

# matrix of features
features, _ = make_blobs(n_samples=1000, n_features=2, random_state=1)

# standart scaler
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features)

# delete one value
true_value = standardized_features[0, 0]
standardized_features[0, 0] = np.nan

# imputer
mean_imputer = Imputer(strategy="mean")
features_mean = mean_imputer.fit_transform(features)

print('True value:', true_value)
print('Imputed value:', features_mean[0, 0])
