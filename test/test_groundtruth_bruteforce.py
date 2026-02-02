import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from utilFuncs import treeUtility


random_seed = 2026
np.random.seed(random_seed)
n_features = 10
x, y = make_regression(1000, n_features=n_features)
model = GradientBoostingRegressor(random_state=random_seed, n_estimators=5, max_depth=5).fit(x, y)

util = treeUtility(model, x[0])
util.evaluate(np.empty(util.n_players, dtype=bool))
semivalue = 0.9

t1 = util.treeprob(semivalue)
print(t1)
t2 = util.groundtruth_bruteforce(semivalue)
print(t2)

print(np.linalg.norm(t1-t2))
