import multiprocessing as mp
from tqdm import tqdm


from .group_testing import group_testing
from .MSR_Banzhaf import MSR_Banzhaf
from .MSR_Prob import MSR_Prob
from .kernelSHAP import kernelSHAP
from .kernelSHAP_unbiased import kernelSHAP_unbiased
from .kernelSHAP_MV import kernelSHAP_MV
from .kernelSHAP_LS import kernelSHAP_LS
from .ARM import ARM
from .linearAME import linearAME
from .GELS import GELS
from .GELS_Shapley import GELS_Shapley
from .SHAP_IQ import SHAP_IQ
from .OFA import OFA
from .linearAppr import linearAppr
from .Adalina import Adalina
from .complement import complement
from .Adalina_All import Adalina_All

_estimators = dict(
    group_testing=group_testing,
    MSR_Banzhaf=MSR_Banzhaf,
    MSR_Prob=MSR_Prob,
    kernelSHAP=kernelSHAP, # only for testing equivalence
    kernelSHAP_unbiased=kernelSHAP_unbiased, # only for testing equivalence
    kernelSHAP_MV=kernelSHAP_MV,
    kernelSHAP_LS=kernelSHAP_LS,
    ARM=ARM,
    linearAME=linearAME,
    GELS=GELS,
    GELS_Shapley=GELS_Shapley,
    SHAP_IQ=SHAP_IQ,
    OFA=OFA,
    linearAppr=linearAppr,
    Adalina=Adalina,
    complement=complement,
    Adalina_All=Adalina_All,
    )


class estimators:
    def __init__(self, n_queries_per_player, n_queries_per_iteration, 
                 n_queries_per_player_per_checkpoint=None, n_processes=1):
        assert n_queries_per_player % n_queries_per_player_per_checkpoint == 0
        
        self.n_queries_per_player = n_queries_per_player
        self.n_queries_per_player_per_checkpoint = n_queries_per_player_per_checkpoint or n_queries_per_player 
        self.n_queries_per_iteration = n_queries_per_iteration
        self.n_processes = n_processes
        
    
    def run(self, util, estimator, semivalue, random_seed_estimator, paired_sampling=0, **kwargs):
        est = _estimators[estimator](util, semivalue,
                                     self.n_queries_per_player, 
                                     self.n_queries_per_player_per_checkpoint,
                                     self.n_queries_per_iteration,
                                     **kwargs)
        
        assert _estimators[estimator].check(semivalue)
        
        if paired_sampling:
            assert _estimators[estimator].can_be_paired(semivalue)
            est.paired_sampling = True
        
        if self.n_processes > 1:
            with mp.Pool(self.n_processes - 1) as pool:
                process = pool.imap(est.process_batch, est.batch_generator(random_seed_estimator))
                for r in tqdm(process, total=-(-est.n_samples // est.batch_size), miniters=self.n_processes-1, maxinterval=float('inf')):
                    est.manage_raw_results(r)
        else:
            for batch in tqdm(est.batch_generator(random_seed_estimator), total=-(-est.n_samples // est.batch_size)):
                est.manage_raw_results(est.process_batch(batch))
        print('\n')
                
        return est.estimate_traj.squeeze()
    
    
def can_estimator_run_semivalue(estimator, semivalue):
    return _estimators[estimator].check(semivalue)


def can_estimator_be_paired(estimator, semivalue):
    return _estimators[estimator].can_be_paired(semivalue)
            