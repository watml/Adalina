from .MSR_Banzhaf import MSR_Banzhaf
from .estimatorTemplate import estimatorTemplate
import numpy as np

# https://arxiv.org/abs/2302.00736 contians its initial version for the Shapley value
# https://openreview.net/forum?id=lvSMIsztka extends it in Appendix E

# inherit MSR_Banzhaf.calculate_estimate
class ARM(MSR_Banzhaf):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        estimatorTemplate.__init__(self, util, semivalue)
        
        assert (n_queries_per_player * util.n_players) % 2 == 0
        self.n_samples = (n_queries_per_player * util.n_players) // 2
        
        assert (n_queries_per_player_per_checkpoint * util.n_players) % 2 == 0
        self.checkpoint_interval = (n_queries_per_player_per_checkpoint * util.n_players) // 2
        
        self.batch_size = -(-n_queries_per_iteration // 2)
        print(f'The number of queries each iteration runs is finalized as {self.batch_size * 2}.')

        self.raw_result_length = 4 * util.n_players
        
        weight = util.compute_cardinality_weights(semivalue)
        tmp = np.arange(1, util.n_players + 1, dtype=np.float64)
        weight_left = np.divide(weight, tmp)
        self.weight_left = weight_left / weight_left.sum()
        weight_right = np.divide(weight, tmp[::-1])
        self.weight_right = weight_right / weight_right.sum()
        self.pool = np.arange(util.n_players + 1)
        
    
    @staticmethod
    def check(semivalue):
        return True
        
    
    def _batch_generator(self, n):
        batch = np.zeros((n, 2, self.util.n_players), dtype=bool)
        for i in range(n):
            s = np.random.choice(self.pool[1:], p=self.weight_left)
            pos_left = np.random.choice(self.pool[:-1], size=s, replace=False)
            s = np.random.choice(self.pool[:-1], p=self.weight_right)
            pos_right = np.random.choice(self.pool[:-1], size=s, replace=False)
            batch[i, 0, pos_left] = True
            batch[i, 1, pos_right] = True        
        return batch
    
    
    def process_each_sample(self, sample, out):
        out = out.reshape(2, 2, self.util.n_players)
        subset = sample[0]
        out[0, 0, subset] = self.util.evaluate(subset)
        out[1, 0, subset] = 1
        subset = sample[1]
        r = self.util.evaluate(subset)
        subset = ~subset
        out[0, 1, subset] = r
        out[1, 1, subset] = 1