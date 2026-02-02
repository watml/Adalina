from .estimatorTemplate import estimatorTemplate
import numpy as np


# inherite check and calculate_estimate
class linearAME(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
        
    
    def _batch_generator(self, n):
        batch = np.empty((n, self.util.n_players + 1), dtype=np.float64)
        for i in range(n):
            if isinstance(self.semivalue, tuple):
                t = np.random.beta(self.semivalue[1], self.semivalue[0])
            else:
                t = self.semivalue
            batch[i, -1] = t
            batch[i, :-1] = np.random.binomial(1, t, size=self.util.n_players)
        return batch
    
    
    def process_each_sample(self, sample, out):
        out[-1] = 1
        t = sample[-1]
        subset = sample[:-1].astype(bool)
        r = self.util.evaluate(subset)
        out = out[:-1]
        out[subset] = r / t
        out[~subset] = -r / (1 - t)
        
    
    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1]