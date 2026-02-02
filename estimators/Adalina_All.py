from .estimatorTemplate import estimatorTemplate
import numpy as np


# inherit check and calculate_estimate
class Adalina_All(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = 2 * util.n_players + 2

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
        out[-2] = r
        out = out[:-2].reshape(2, self.util.n_players)
        if size:     
            out[:, sample] = self.weights[size-1] / size
        if self.util.n_players - size:
            out[:, ~sample] = -self.weights[size] / (self.util.n_players - size)
        out *= self.util.n_players / self.sampling_prob[size]
        out[1] *= r
        
        
        
    def calculate_estimate(self):
        count = self.summed_result[-1]
        quo = self.summed_result[self.util.n_players:-1] / count
        return quo[:-1] - self.summed_result[:self.util.n_players] / count * quo[-1]
    
        
    