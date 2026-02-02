from .estimatorTemplate import estimatorTemplate
import numpy as np

# it is equal to kernelSHAP_LS using scalar_vr = 'diff' and sampling = 'kernel'
# it is merely used to verify that they are equal
class kernelSHAP(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.summed_result_length = util.n_players * (util.n_players + 1) + 1
        
        self.raw_result_length = util.n_players + 1
            
        tmp = np.arange(1, util.n_players, dtype=np.float64)
        tmp = 1 / np.multiply(tmp, tmp[::-1])
        self.sampling_prob = tmp / tmp.sum()
        
        self.grand_util, self.empty_util = self.compute_extreme_util()
        
        self.pool = np.arange(util.n_players)

    
    @staticmethod
    def check(semivalue):
        return semivalue == (1, 1)
        
    
    @staticmethod
    def can_be_paired(semivalue):
        if semivalue == (1, 1):
            return True
        else:
            return False
        
    
    def _batch_generator(self, n):
        batch = np.zeros((n, self.util.n_players), dtype=bool)
        for i in range(n):      
            s = np.random.choice(self.pool[1:], p=self.sampling_prob)
            pos = np.random.choice(self.pool, size=s, replace=False)
            batch[i, pos] = True
        return batch
    
    
    def process_batch(self, batch):
        raw_results = np.empty((len(batch), self.raw_result_length), dtype=np.float64)
        raw_results[:, :-1] = batch
        for i, sample in enumerate(batch):
            raw_results[i, -1] = self.util.evaluate(sample)
        return raw_results
    
    
    def sumup(self, raw_results):
        subsets = raw_results[:, :-1]
        queried_results = raw_results[:, [-1]] - self.empty_util
        
        self.summed_result[-1] += len(subsets)
        r = self.summed_result[:-1].reshape(self.util.n_players + 1, self.util.n_players)
        r[:-1] += subsets.T @ subsets
        r[-1] += (subsets * queried_results).sum(axis=0)


    def calculate_estimate(self):
        count = self.summed_result[-1]
        r = self.summed_result[:-1].reshape(self.util.n_players + 1, self.util.n_players)
        A_inv = np.linalg.pinv(r[:-1] / count)
        vec_b = r[-1] / count
        vec_1 = np.ones(len(vec_b))
        tmp = vec_b - (np.dot(vec_1, np.dot(A_inv, vec_b)) - self.grand_util + self.empty_util) / np.dot(vec_1, np.dot(A_inv, vec_1))
        return np.dot(A_inv, tmp)