import numpy as np


class estimatorTemplate:
    def __init__(self, util, semivalue):
        util.check_semivalue(semivalue)
        self.util = util
        self.semivalue = semivalue
        self.paired_sampling = False
        
        # used in manage_raw_results
        self.pos_buffer = 0 
        self.pos_traj = 0
        
        # defined in init_at_start
        self.estimate_traj = None
        self.raw_result_buffer = None
        self.summed_result = None
        
        # to be overridden
        self.n_samples = None
        self.checkpoint_interval = None
        self.batch_size = None
        self.raw_result_length = None
        self.summed_result_length = None # if it is None, it is set to be raw_result_length
        self.n_players = None # to change the size of samples in batch_generator, e.g., group_testing needs it.

        
    def batch_generator(self, randomseed):
        self.init_at_start(randomseed)
        
        if self.paired_sampling:
            assert self.n_samples % 2 == 0 and self.batch_size % 2 == 0
            
            n_samples, batch_size = self.n_samples // 2, self.batch_size // 2
            n_batches, n_remaining_samples = np.divmod(n_samples, batch_size)
            # assume that each sample corresponds to a subset
            samples = np.empty((self.batch_size, self.n_players or self.util.n_players), dtype=bool)
            for _ in range(n_batches):
                gen = self._batch_generator(batch_size)
                samples[0::2, :] = gen
                samples[1::2, :] = ~gen
                yield samples
            
            if n_remaining_samples:
                samples = np.empty((n_remaining_samples*2, self.util.n_players), dtype=bool)
                gen = self._batch_generator(n_remaining_samples)
                samples[0::2, :] = gen
                samples[1::2, :] = ~gen
                yield samples
        else:        
            n_batches, n_remaining_samples = np.divmod(self.n_samples, self.batch_size)
            for _ in range(n_batches):
                yield self._batch_generator(self.batch_size)
                
            if n_remaining_samples:
                yield self._batch_generator(n_remaining_samples)
            
            
    def init_at_start(self, randomseed):
        assert self.n_samples % self.checkpoint_interval == 0
        
        np.random.seed(randomseed)
        shape = (self.n_samples//self.checkpoint_interval, self.util.n_players)
        self.estimate_traj = np.empty(shape, dtype=np.float64)
        self.raw_result_buffer = np.empty((self.batch_size+self.checkpoint_interval-1, self.raw_result_length), dtype=np.float64)
        self.summed_result = np.zeros(self.summed_result_length or self.raw_result_length, dtype=np.float64)
            
    
    def manage_raw_results(self, raw_results):
        n = len(raw_results)
        self.raw_result_buffer[self.pos_buffer : self.pos_buffer+n] = raw_results
        self.pos_buffer += n
        n_processed = self.pos_buffer // self.checkpoint_interval
        if n_processed:
            for i in range(n_processed):
                self.sumup(self.raw_result_buffer[i*self.checkpoint_interval : (i+1)*self.checkpoint_interval])
                self.estimate_traj[self.pos_traj] = self.calculate_estimate()
                self.pos_traj += 1
                
            n_remaining = self.pos_buffer - n_processed * self.checkpoint_interval
            self.raw_result_buffer[:n_remaining] = self.raw_result_buffer[n_processed * self.checkpoint_interval : self.pos_buffer]
            self.pos_buffer = n_remaining 
            
    
    def process_batch(self, batch):
        raw_results = np.zeros((len(batch), self.raw_result_length), dtype=np.float64)
        for i, sample in enumerate(batch):
            self.process_each_sample(sample, raw_results[i])
        return raw_results
    
    
    def sumup(self, raw_results):
        self.summed_result += raw_results.sum(axis=0)
        
    
    def compute_extreme_util(self):
        subset = np.ones(self.util.n_players, dtype=bool)
        grand_util = self.util.evaluate(subset)
        subset.fill(0)
        empty_util = self.util.evaluate(subset)
        return grand_util, empty_util
            
    
    @staticmethod
    def check(semivalue):
        # return True if the estimator can approximate the specified semivalue and False otherwise
        return True
    
    
    @staticmethod
    def can_be_paired(semivalue):
        # return True if paired sampling can benefit the estimator
        return False

            
    def _batch_generator(self, n):
        # to be defined, which generates n samples
        return 0

    
    def process_each_sample(self, sample, out):
        # to be defined  
        pass
   
    
    def calculate_estimate(self):
        # to be defined
        return 0