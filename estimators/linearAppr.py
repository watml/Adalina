from .kernelSHAP_MV import kernelSHAP_MV
from .estimatorTemplate import estimatorTemplate
import numpy as np
import numbers

# inherit _batch_generator
class linearAppr(kernelSHAP_MV):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration,
                 aux='zero'):
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
        
        self.pool = np.arange(util.n_players)
        weights = util.compute_cardinality_weights(semivalue)
        tmp = np.arange(1, util.n_players, dtype=np.float64)
        self.sampling_prob = weights[:-1] ** 2 / tmp
        self.sampling_prob += weights[1:] ** 2 / tmp[::-1]
        self.sampling_prob **= 0.5
        self.sampling_prob /= self.sampling_prob.sum()
        
        self.weights = util.compute_cardinality_weights(semivalue)
        grand_util, empty_util = self.compute_extreme_util()
        
        if aux == 'zero':
            self.aux = 0
        elif aux == 'default':
            self.aux = (grand_util + empty_util) / 2
        elif aux == 'empty':
            self.aux = empty_util
        else:
            assert isinstance(aux, numbers.Number)
            self.aux = aux
                   
        self.shift = (grand_util - self.aux) * weights[-1] -  (empty_util - self.aux) * weights[0]
        
    
    @staticmethod
    def check(semivalue):
        return True
    
    
    @staticmethod
    def can_be_paired(semivalue):
        if isinstance(semivalue, tuple):
            if semivalue[0] == semivalue[1]:
                return True
            else:
                return False
        else:
            if semivalue == 0.5:
                return True
            else:
                return False
    
    
    def process_each_sample(self, sample, out):
        out[-1] = 1
        size = sample.sum()
        out = out[:-1]
        out[sample] = self.weights[size-1] / size
        out[~sample] = -self.weights[size] / (self.util.n_players - size)
        out *= self.util.n_players * (self.util.evaluate(sample) - self.aux) / self.sampling_prob[size-1]
    
    
    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1] + self.shift
        
        
    