from sklearn import datasets, metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# mnist dataset
digits = datasets.load_digits()

# load features and target data
features = digits.data
target = digits.target

# scaler
scaler = StandardScaler()

# LR object & pipeline
logit = LogisticRegression()
pipeline = make_pipeline(scaler, logit)

# k-fold validation
k_fold =KFold(n_splits=10, shuffle=True, random_state=1)

cv_results = cross_val_score(pipeline, features, target, cv=k_fold, scoring="accuracy", n_jobs=-1)

print('Mean value:', cv_results.mean())