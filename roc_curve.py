import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# matrix of features and target vector
features, target = make_classification(n_samples=10000, n_features=10,
                                       n_classes=2, n_informative=3,
                                       random_state=3)

# training and test split
train, test, target_train, target_test = train_test_split(features, target,
                                                          test_size=0.1,
                                                          random_state=1)

# logistic classificator
logit = LogisticRegression()
logit.fit(train, target_train)

# predict
target_probabilities = logit.predict_proba(test)[:, 1]

# outcomes
f_positive, t_positive, treshold = roc_curve(target_test, target_probabilities)

# plot
plt.title('ROC curve')
plt.plot(f_positive, t_positive)
plt.plot([0, 1], ls='--')
plt.plot([0, 0], [1, 0], c='.7')
plt.plot([1, 1], c='.7')
plt.xlabel('TP')
plt.plot('FP')
plt.show()
