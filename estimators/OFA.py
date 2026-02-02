from .kernelSHAP_LS import kernelSHAP_LS
from .estimatorTemplate import estimatorTemplate
import numpy as np

# https://cs.uwaterloo.ca/~y328yu/publication/li-yu-24-b/li-yu-24-b.pdf
# we chose to approximate phi_{i,1}^{+} and phi_{i,1}^{-}

# inherite _batch_generator and process_batch
class OFA(kernelSHAP_LS):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
        
        self.summed_result_length = 4 * util.n_players * (util.n_players - 1)
        
        self.pool = np.arange(util.n_players)
        weights = util.compute_cardinality_weights(semivalue)
        tmp = np.arange(1, util.n_players, dtype=np.float64)
        self.sampling_prob = weights[:-1] ** 2 / tmp
        self.sampling_prob += weights[1:] ** 2 / tmp[::-1]
        self.sampling_prob **= 0.5
        self.sampling_prob /= self.sampling_prob.sum()
        
        self.weights = util.compute_cardinality_weights(semivalue)
        grand_util, empty_util = self.compute_extreme_util()
        self.shift = grand_util * weights[-1] -  empty_util * weights[0]
        
    
    @staticmethod
    def check(semivalue):
        return True
    
    
    @staticmethod
    def can_be_paired(semivalue):
        return False
        
    
    def sumup(self, raw_results):
        n = len(raw_results)
        subsets = raw_results[:, :-1].astype(np.int64)
        queried_results = raw_results[:, [-1]]
        sizes = subsets.sum(axis=1)
        tmp = np.zeros((n, self.util.n_players - 1), dtype=np.int64)
        tmp[np.arange(n), sizes-1] = 1
        
        r = self.summed_result.reshape(2, 2, self.util.n_players, self.util.n_players - 1)
        r[0, 0] += (subsets * queried_results).T @ tmp
        r[1, 0] += subsets.T @ tmp
        
        subsets = 1 - subsets
        r[0, 1] += (subsets * queried_results).T @ tmp
        r[1, 1] += subsets.T @ tmp
        
        
    
    def calculate_estimate(self):
        r = self.summed_result.reshape(2, 2, self.util.n_players, self.util.n_players - 1)
        count = r[1].copy()
        count[count == 0] = -1
        quo = r[0] / count
        return quo[0] @ self.weights[:-1] - quo[1] @ self.weights[1:] + self.shift