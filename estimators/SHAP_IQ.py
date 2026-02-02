from .kernelSHAP_MV import kernelSHAP_MV
from .estimatorTemplate import estimatorTemplate
import numpy as np

# https://arxiv.org/pdf/2303.01179

# inherite kernelSHAP._batch_generator
class SHAP_IQ(kernelSHAP_MV):
    def __init__(self, util, param,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, param)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
            
        tmp = 1 / np.arange(1, util.n_players, dtype=np.float64)
        self.scalar = 2 * tmp.sum()
        tmp = tmp * tmp[::-1]
        self.sampling_prob = tmp / tmp.sum()
        self.pool = np.arange(util.n_players)
        grand_util, self.empty_util = self.compute_extreme_util()
        weights = util.compute_cardinality_weights(param)
        self.constant = (grand_util - self.empty_util) * weights[-1]
        self.weights_p = self.pool[::-1] * weights
        self.weights_n = self.pool * weights
        
    
    @staticmethod
    def check(semivalue):
        return True
    
    
    @staticmethod
    def can_be_paired(semivalue):
        return False
    
    
    def process_each_sample(self, sample, out):
        r = self.util.evaluate(sample)
        size = sample.sum()
        out[-1] = 1
        out = out[:-1]
        out[sample] = self.weights_p[size-1] * (r - self.empty_util)
        out[~sample] = -self.weights_n[size] * (r - self.empty_util)

    
    def calculate_estimate(self):
        return self.constant + self.summed_result[:-1] / self.summed_result[-1] * self.scalar