from .estimatorTemplate import estimatorTemplate
import numpy as np
from scipy.special import comb

# Eq. (3) in https://arxiv.org/pdf/2506.11849

# inherit check and calculate_estimate
class MSR_Witter(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1

        # compute something required
        self.subset_weights = np.zeros(util.n_players + 1, dtype=np.float64)
        self.subset_weights[:-1] = util.compute_weights(semivalue)
        weights_squared = self.subset_weights[:-1] ** 2
        sampling_prob = np.zeros(util.n_players+1, dtype=np.float64)
        tmp = np.arange(1, util.n_players+1, dtype=np.float64) / util.n_players
        sampling_prob[:-1] += weights_squared * tmp[::-1]
        sampling_prob[1:] += weights_squared * tmp
        self.sampling_weights = np.sqrt(sampling_prob)
        self.sampling_prob = self.sampling_weights * comb(util.n_players, np.arange(util.n_players+1))
        tmp = self.sampling_prob.sum()
        self.sampling_prob /= tmp
        self.sampling_weights /= tmp
        self.pool = np.arange(util.n_players+1)
        
    
    def _batch_generator(self, n):
        batch = np.zeros((n, self.util.n_players), dtype=bool)
        for i in range(n):
            s = np.random.choice(self.pool, p=self.sampling_prob)
            pos = np.random.choice(self.pool[:-1], size=s, replace=False)
            batch[i, pos] = True
        return batch
    
    
    def process_each_sample(self, sample, out):
        r = self.util.evaluate(sample)
        size = sample.sum()
        out[-1] = 1
        out = out[:-1]
        out[sample] = self.subset_weights[size-1] * r / self.sampling_weights[size]
        out[~sample] = -self.subset_weights[size] * r / self.sampling_weights[size]
        
        
    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1]