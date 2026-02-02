from .estimatorTemplate import estimatorTemplate
import numpy as np


class kernelSHAP_MV(estimatorTemplate):
    def __init__(self, util, semivalue,
                 n_queries_per_player, n_queries_per_player_per_checkpoint, 
                 n_queries_per_iteration,
                 scalar_vr='diff',
                 sampling='geomean'):
        # vr = variance reduction
        super().__init__(util, semivalue)
        
        self.n_samples = n_queries_per_player * util.n_players
        
        self.checkpoint_interval = n_queries_per_player_per_checkpoint * util.n_players
        
        self.batch_size = n_queries_per_iteration

        self.raw_result_length = util.n_players + 1
        
        grand_util, empty_util = self.compute_extreme_util()
        self.offset = (grand_util - empty_util) / util.n_players
        self.pool = np.arange(util.n_players)
        
        if scalar_vr == 'zero':
            self.scalar_vr = 0
        elif scalar_vr == 'diff':
            self.scalar_vr = self.offset
        else:
            raise ValueError('Check scalar_vr.')
            
        if sampling == 'kernel':
            tmp = np.arange(1, util.n_players, dtype=np.float64)
            tmp = 1 / np.multiply(tmp, tmp[::-1])
            self.sampling_prob = tmp / tmp.sum()
            self.scalars = np.full(util.n_players - 1, 
                                   2 * np.reciprocal(np.arange(1, util.n_players, dtype=np.float64)).sum(), 
                                   np.float64)
        elif sampling == 'leverage':
            tmp = np.reciprocal(np.arange(1, util.n_players, dtype=np.float64))
            self.sampling_prob = np.full(util.n_players - 1, 1 / (util.n_players-1), dtype=np.float64)
            self.scalars = (tmp * util.n_players) * (tmp[::-1] * (util.n_players - 1))        
        elif sampling == 'geomean':
            tmp = np.arange(1, util.n_players, dtype=np.float64) ** 0.5
            self.scalars = 1 / (tmp * tmp[::-1])
            tmp_sum = self.scalars.sum()
            self.sampling_prob = self.scalars / tmp_sum
            self.scalars *= tmp_sum * util.n_players
        else:
            raise ValueError('Check sampling.')
            
    
    @staticmethod
    def check(semivalue):
        return semivalue == (1, 1)
    
    
    @staticmethod
    def can_be_paired(semivalue):
        # it produces worst performance.
        # note that this has been inheritted.
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
    
    
    def process_each_sample(self, sample, out):
        out[-1] = 1
        size = sample.sum()
        out[:-1] = (sample - size / self.util.n_players) * (self.util.evaluate(sample) - self.scalar_vr * size)
        out[:-1] *= self.scalars[size-1]
    
        
    def calculate_estimate(self):
        return self.summed_result[:-1] / self.summed_result[-1] + self.offset