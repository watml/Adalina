from .estimatorTemplate import estimatorTemplate
import numpy as np

# Eq. (3) in https://arxiv.org/pdf/2506.11849

class MSR_Prob(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1

        # compute something required
        self.weights = util.compute_cardinality_weights(semivalue)
        weights_squared = self.weights ** 2
        self.sampling_prob = np.zeros(util.n_players+1, dtype=np.float64)
        tmp = np.arange(1, util.n_players+1, dtype=np.float64)
        self.sampling_prob[:-1] += weights_squared / tmp[::-1]
        self.sampling_prob[1:] += weights_squared / tmp
        self.sampling_prob **= 0.5
        self.sampling_prob /= self.sampling_prob.sum()
        
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
        scalar = r * self.util.n_players / self.sampling_prob[size]
        if size:
            out[sample] = self.weights[size-1] / size * scalar
        if self.util.n_players - size:
            out[~sample] = -self.weights[size] / (self.util.n_players - size) * scalar
        
        
    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1]