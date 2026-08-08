from .estimatorTemplate import estimatorTemplate
import numpy as np

# Eq. (4) in https://openreview.net/pdf?id=u359tNBpxF

class MSR_Banzhaf(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration
        
        self.raw_result_length = 4 * util.n_players

    
    @staticmethod
    def check(semivalue):
        return not isinstance(semivalue, tuple)
        
    
    def _batch_generator(self, n):
        return np.random.binomial(1, self.semivalue, size=(n, self.util.n_players)).astype(bool)  
    
    
    def process_each_sample(self, sample, out):
        r = self.util.evaluate(sample)
        out = out.reshape(2, 2, self.util.n_players)
        out[0, 0, sample] = r
        out[1, 0, sample] = 1
        sample = ~sample
        out[0, 1, sample] = r
        out[1, 1, sample] = 1
            

    def calculate_estimate(self):
        r = self.summed_result.reshape(2, 2, self.util.n_players)
        sumup = r[0]
        count = r[1].copy()
        count[count == 0] = -1
        quo = np.divide(sumup, count)
        return quo[0] - quo[1]