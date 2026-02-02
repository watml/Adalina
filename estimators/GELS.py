from .estimatorTemplate import estimatorTemplate
import numpy as np




class GELS(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration,):
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = 2 * (util.n_players + 1)
            
        weights = util.compute_cardinality_weights(semivalue)
        self.scalar = (np.divide(weights, np.arange(util.n_players, 0, -1)) * util.n_players).sum()
        tmp = np.arange(1, util.n_players + 1, dtype=np.float64)
        tmp = np.multiply(tmp / (util.n_players + 1), (util.n_players + 1 - tmp) / util.n_players)
        tmp = np.reciprocal(tmp)
        weights = np.multiply(weights, tmp)
        self.sampling_prob = weights / weights.sum()
        self.pool = np.arange(util.n_players + 1)
        
    
    def _batch_generator(self, n):
        batch = np.zeros((n, self.util.n_players + 1), dtype=bool)
        for i in range(n):      
            s = np.random.choice(self.pool[1:], p=self.sampling_prob)
            pos = np.random.choice(self.pool, size=s, replace=False)
            batch[i, pos] = True
        return batch


    def process_each_sample(self, sample, out):
        out = out.reshape(2, self.util.n_players + 1)
        out[0, sample] = self.util.evaluate(sample[:-1])
        out[1, sample] = 1
    
    
    def calculate_estimate(self):
        r = self.summed_result.reshape(2, self.util.n_players + 1)
        count = r[1].copy()
        count[count == 0] = -1
        tmp = np.divide(r[0], count)
        tmp *= self.scalar
        return tmp[:-1] - tmp[-1]