from .estimatorTemplate import estimatorTemplate
import numpy as np



class complement(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration):
        super().__init__(util, semivalue)
        
        assert (n_queries_per_player * util.n_players) % 2 == 0
        self.n_samples = (n_queries_per_player * util.n_players) // 2
        
        assert (n_queries_per_player_per_checkpoint * util.n_players) % 2 == 0
        self.checkpoint_interval = (n_queries_per_player_per_checkpoint * util.n_players) // 2
        
        self.batch_size = -(-n_queries_per_iteration // 2)
        print(f'The number of queries each iteration runs is finalized as {self.batch_size * 2}.')

        self.raw_result_length = util.n_players + 1
        
        self.summed_result_length = 2 * util.n_players ** 2 
        
        self.pool = np.arange(1, util.n_players + 1)
        
    
    @staticmethod
    def check(semivalue):
        return semivalue == (1, 1)
        
        
    def _batch_generator(self, n):
        batch = np.zeros((n, self.util.n_players), dtype=bool)
        for i in range(n):
            s = np.random.choice(self.pool)
            pi = np.random.permutation(self.util.n_players)
            batch[i, pi[:s]] = True
            # Note what in the above is equal to
            # pos = np.random.choice(np.arange(self.util.n_players), size=s, replace=False)
            # batch[i, pos] = True
            # But we stay loyal to the original paper
        return batch
    
    
    def process_batch(self, batch):
        raw_results = np.zeros((len(batch), self.raw_result_length), dtype=np.float64)
        raw_results[:, :-1] = batch
        for i, sample in enumerate(batch):
            raw_results[i, -1] = self.util.evaluate(sample) - self.util.evaluate(~sample)
        # when the size of raw_results is large, the cost of communication would significantly hurt the performance
        return raw_results
        
    
    def sumup(self, raw_results):
        n = len(raw_results)
        subsets = raw_results[:, :-1].astype(np.int64)
        queried_results = raw_results[:, [-1]]
        tmp = np.zeros_like(subsets)
        sizes = subsets.sum(axis=1)
        tmp[np.arange(n), sizes-1] = 1
        
        r = self.summed_result.reshape(2, self.util.n_players, self.util.n_players)
        r[0] += (subsets * queried_results).T @ tmp
        r[1] += subsets.T @ tmp
        
        subsets = 1 - subsets
        tmp = np.zeros_like(subsets)
        sizes = self.util.n_players - sizes
        tmp[np.arange(n), sizes-1] = 1
        
        r[0] -= (subsets * queried_results).T @ tmp
        r[1] += subsets.T @ tmp
        
    
    def calculate_estimate(self):
        r = self.summed_result.reshape(2, self.util.n_players, self.util.n_players)
        count = r[1].copy()
        count[count == 0] = -1
        return np.mean(r[0] / count, axis=1)