from .GELS import GELS
from .estimatorTemplate import estimatorTemplate
import numpy as np

# Algorithm 2 in https://arxiv.org/pdf/2302.11431

class group_testing(GELS):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)

        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
        
        self.n_players = util.n_players + 1
        
        self.pool = np.arange(util.n_players + 1)
        tmp = 1 / np.arange(1, util.n_players + 1, dtype=np.float64)
        weights = tmp + tmp[::-1]
        self.const = weights.sum()
        self.sampling_prob = weights / self.const
        
    
    @staticmethod
    def check(semivalue):
        return semivalue == (1, 1)
    
    
    @staticmethod
    def can_be_paired(semivalue):
        if semivalue == (1, 1):
            return True
        else:
            return False
    
    
    def process_each_sample(self, sample, out):
        out[-1] = 1
        r = self.util.evaluate(sample[:-1]) * sample
        out[:-1] = r[:-1] - r[-1]
        out[:-1] *= self.const
        

    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1]