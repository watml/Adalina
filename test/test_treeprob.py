import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from utilFuncs import treeUtility

semivalue=(1, 1)

# test regression
random_seed = 2026
np.random.seed(random_seed)
n_features = 10
x, y = make_regression(1000, n_features=n_features)
model = GradientBoostingRegressor(random_state=random_seed, n_estimators=5, max_depth=5).fit(x, y)
util = treeUtility(model, x[0])
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))

model = DecisionTreeRegressor(random_state=random_seed,max_depth=5).fit(x, y)
util = treeUtility(model, x[0])
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))


# test binary classification
x, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=2,
    random_state=0
)
model = GradientBoostingClassifier(random_state=random_seed, n_estimators=5, max_depth=5).fit(x, y)
util = treeUtility(model, x[0], 0)
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))


model = DecisionTreeClassifier(random_state=random_seed,max_depth=5).fit(x, y)
util = treeUtility(model, x[0], 0)
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))



# test multiclass classification
x, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_classes=4,
    n_clusters_per_class=1,
    random_state=0
)
model = GradientBoostingClassifier(random_state=random_seed, n_estimators=5, max_depth=5).fit(x, y)
util = treeUtility(model, x[0], 3)
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))


model = DecisionTreeClassifier(random_state=random_seed,max_depth=5).fit(x, y)
util = treeUtility(model, x[0], 3)
t1 = util.treeprob(semivalue)
t2 = util.groundtruth_bruteforce(semivalue)
print(np.linalg.norm(t1-t2))







