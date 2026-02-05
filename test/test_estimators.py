import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from estimators import estimators
import numpy as np
from sklearn.datasets import make_regression
from createTreeModel import createTreeModel
from sklearn.ensemble import GradientBoostingRegressor
from utilFuncs import treeUtility

n_processes = 1
estimator = 'Adalina_All'
semivalue = (1,1)
paired_sampling = 0
n_players = 10
n_queries_per_player = 500
n_queries_per_iteration = 100
n_queries_per_player_per_checkpoint = 50
kwargs = dict()
#kwargs.update(scalar_vr='zero', sampling='kernel')
#kwargs.update(unbiased=0)
#kwargs.update(aux='default')

print(estimator)
print(semivalue)
print(paired_sampling)
print(kwargs)



# =============================================================================
# random_seed = 2026
# np.random.seed(random_seed)
# x, y = make_regression(1000, n_features=n_players)
# y = np.abs(y)
# model = GradientBoostingRegressor(random_state=random_seed, n_estimators=5, max_depth=5).fit(x, y)
# util = treeUtility(model, x[0])
# =============================================================================

# 44, 1475, 43174
model, X_test, y_test = createTreeModel(43174, 2, 2026)
util = treeUtility(model, X_test[0])


est = estimators(n_queries_per_player, n_queries_per_iteration, 
                 n_queries_per_player_per_checkpoint, n_processes)

ground_truth = util.treeprob(semivalue)

print(ground_truth)

estimate_traj = est.run(util, estimator, semivalue, 2026, paired_sampling, **kwargs)

print(estimate_traj[-1])

print(np.linalg.norm(estimate_traj - ground_truth[None, :], axis=1))