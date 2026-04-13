from .linearAppr import linearAppr
from .estimatorTemplate import estimatorTemplate
import numpy as np


class Adalina(linearAppr):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration
        
        self.pool = np.arange(util.n_players)
        weights = util.compute_cardinality_weights(semivalue)
        tmp = np.arange(1, util.n_players, dtype=np.float64)
        self.sampling_prob = weights[:-1] ** 2 / tmp
        self.sampling_prob += weights[1:] ** 2 / tmp[::-1]
        self.sampling_prob **= 0.5
        self.sampling_prob /= self.sampling_prob.sum()
        
        self.weights = util.compute_cardinality_weights(semivalue)
        self.grand_util, self.empty_util = self.compute_extreme_util()
    
        self.raw_result_length = 2 * util.n_players + 2

        
    @staticmethod
    def can_be_paired(semivalue):
        return False
        
        
    def process_each_sample(self, sample, out):
        r = self.util.evaluate(sample)
        size = sample.sum()
        out[-1] = 1
        out[-2] = r
        out = out[:-2].reshape(2, self.util.n_players)
        out[:, sample] = self.weights[size-1] / size
        out[:, ~sample] = -self.weights[size] / (self.util.n_players - size)
        out *= self.util.n_players / self.sampling_prob[size-1]
        out[1] *= r
        
    
    def calculate_estimate(self):
        count = self.summed_result[-1]
        quo = self.summed_result[:-1] / count
        shift = (self.grand_util - quo[-1]) * self.weights[-1] - (self.empty_util - quo[-1]) * self.weights[0]
        return quo[self.util.n_players:-1] - quo[:self.util.n_players] * quo[-1] + shift