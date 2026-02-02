from .kernelSHAP_MV import kernelSHAP_MV
import numpy as np

# Sketched Regression in https://arxiv.org/pdf/2506.05216

class kernelSHAP_LS(kernelSHAP_MV):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration,
                 scalar_vr='diff',
                 sampling='geomean'):
        # vr = variance reduction
        super().__init__(util, semivalue,
                         n_queries_per_player, n_queries_per_player_per_checkpoint, 
                         n_queries_per_iteration,
                         scalar_vr,
                         sampling)
        
        
        self.raw_result_length = util.n_players + 1
        
        self.summed_result_length = util.n_players * (util.n_players + 1) + 1
        
        self.Q = np.zeros((util.n_players - 1, util.n_players), dtype=np.float64)
        for i in range(util.n_players - 1):
            self.Q[i, :i+1] = 1
            self.Q[i, i+1] = -i-1
            self.Q[i] /= np.sqrt((i+1)*(i+2))
            
    
    def process_batch(self, batch):
        raw_results = np.zeros((len(batch), self.raw_result_length), dtype=np.float64)
        raw_results[:, :-1] = batch
        for i, sample in enumerate(batch):
            raw_results[i, -1] = self.util.evaluate(sample)
        # when the size of raw_results is large, the cost of communication would significantly hurt the performance
        # e.g., (100, 300, 300)
        return raw_results
    
    
    def sumup(self, raw_results):
        subsets = raw_results[:, :-1]
        queried_results = raw_results[:, [-1]]
        self.summed_result[-1] += len(queried_results)
        sizes = subsets.sum(axis=1, keepdims=True)
        
        r = self.summed_result[:-1].reshape(self.util.n_players + 1, self.util.n_players)
        r[:-1] += subsets.T @ subsets
        r[-1] += (subsets * (queried_results - self.scalar_vr * sizes)).sum(axis=0)


    def calculate_estimate(self):
        count = self.summed_result[-1]
        r = self.summed_result[:-1].reshape(self.util.n_players + 1, self.util.n_players) / count
        return self.Q.T @ np.linalg.pinv(self.Q @ r[:-1] @ self.Q.T) @ self.Q @ r[-1] + self.offset   