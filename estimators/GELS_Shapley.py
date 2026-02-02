from .GELS import GELS
from .estimatorTemplate import estimatorTemplate
import numpy as np


# inherit process_batch
class GELS_Shapley(GELS):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration,):
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration
        
        self.raw_result_length = 2 * util.n_players
        
        tmp = 1 / np.arange(1, util.n_players, dtype=np.float64)
        self.scalar = tmp.sum()
        weights = np.multiply(tmp, tmp[::-1])
        self.sampling_prob = weights / weights.sum()
        self.pool = np.arange(util.n_players)
        
        grand_util, empty_util = self.compute_extreme_util()
        self.const = grand_util - empty_util
        
    
    @staticmethod
    def check(semivalue):
        return semivalue == (1, 1)
        
    
    def _batch_generator(self, n):
        batch = np.zeros((n, self.util.n_players), dtype=bool)
        for i in range(n):      
            s = np.random.choice(self.pool[1:], p=self.sampling_prob)
            pos = np.random.choice(self.pool, size=s, replace=False)
            batch[i, pos] = True
        return batch
        
    
    def process_each_sample(self, sample, out):
        out = out.reshape(2, self.util.n_players)
        out[0, sample] = self.util.evaluate(sample)
        out[1, sample] = 1
        
        
    def calculate_estimate(self):
        r = self.summed_result.reshape(2, self.util.n_players)
        count = r[1].copy()
        count[count == 0] = -1
        estimate = np.divide(r[0], count)
        estimate *= self.scalar
        offset = (self.const - estimate.sum()) / self.util.n_players
        return estimate + offset