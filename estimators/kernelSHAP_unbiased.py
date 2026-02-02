from .estimatorTemplate import estimatorTemplate
from .kernelSHAP import kernelSHAP
import numpy as np

# Eq. (9) in https://arxiv.org/pdf/2012.01536

# it is equal to kernelSHAP_MV using scalar_vr = 'zero' and sampling = 'kernel'
# it is merely used to verify that they are equal
class kernelSHAP_unbiased(kernelSHAP):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        # vr = variance reduction
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.summed_result_length = util.n_players + 1
        
        self.raw_result_length = util.n_players
        
        self.grand_util, self.empty_util = self.compute_extreme_util()
        
            
        tmp = np.arange(1, util.n_players, dtype=np.float64)
        tmp = 1 / np.multiply(tmp, tmp[::-1])
        self.sampling_prob = tmp / tmp.sum()
        
        self.pool = np.arange(util.n_players)
        
        u1 = .5
        tmp_ = np.arange(1, util.n_players-1)
        u2 = (tmp_ / tmp_[::-1]).sum() / tmp.sum() / util.n_players / (util.n_players-1)
        self.A_inv = np.full((util.n_players, util.n_players), -u2/(u1-u2)/(u1+u2*(util.n_players-1)), dtype=np.float64)
        np.fill_diagonal(self.A_inv, (u1+u2*(util.n_players-2))/(u1-u2)/(u1+u2*(util.n_players-1)))
        
    
    def process_batch(self, batch):
        raw_results = np.empty((len(batch), self.raw_result_length), dtype=np.float64)
        for i, sample in enumerate(batch):
            raw_results[i] = sample * self.util.evaluate(sample) - self.empty_util / 2
        return raw_results
    
    
    def sumup(self, raw_results):
        self.summed_result[-1] += len(raw_results)
        self.summed_result[:-1] += raw_results.sum(axis=0)
        
        
    def calculate_estimate(self):
        count = self.summed_result[-1]
        vec_b = self.summed_result[:-1] / count
        vec_1 = np.ones(len(vec_b))
        tmp = vec_b - (np.dot(vec_1, np.dot(self.A_inv, vec_b)) - self.grand_util + self.empty_util) / np.dot(vec_1, np.dot(self.A_inv, vec_1))
        return np.dot(self.A_inv, tmp)