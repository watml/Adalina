import os
# If there are n cpus, without the following specification, each process would
# create n threads. So, given n_processes = n, there would be nxn threads in total,
# which could hurt performance. Make sure n_processes x n_threads <= n_cpus.
# it should be done before importing any other modules.
NUM_THREAD = 1
os.environ["OMP_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["MKL_NUM_THREADS"] = f"{NUM_THREAD}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{NUM_THREAD}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{NUM_THREAD}"
import numpy as np

arg_dict = dict(
    # fixed
    root='exp',
    n_queries_per_iteration=100,
    random_seed_anythingelse=2026,
    # varied
    n_estimators=[0, 10, 50],
    n_queries_per_player=[20, 100, 1000],
    n_queries_per_player_per_checkpoint=[2, 10],
    dataset_id=[4538, 44, 43174, 1475, 41150, 41145, 41168, 44975, 4549, 1478, 40996, 40926],
    semivalue=np.arange(.1, 1, .1).round(1).tolist() + \
        [(16, 1), (4, 1), (1, 1), (1, 4), (1, 16), (16, 4), (2, 2), (8, 8), (4, 16)],
    random_seed_estimator=range(10),
    paired_sampling=[0, 1],
    estimator=dict(
        estimator=['MSR_Banzhaf', 'MSR_Prob', 'ARM', 'linearAME', 'GELS', 
                   'GELS_Shapley', 'SHAP_IQ', 'Adalina_All', 'Adalina'],
        linearAppr=dict(
            aux=['default', 'empty']
            ),
        kernelSHAP_MV=dict(
            scalar_vr=['diff'],
            sampling=['kernel', 'leverage', 'geomean'],
            ),
        ),       
    )
    

def skip_arg(arg):
    if arg['n_estimators'] in [0, 50]:
        if arg['dataset_id'] not in [44, 1475, 41145, 41150]:
            return 1
        if arg['estimator'] != 'kernelSHAP_MV':
            return 1
        if arg['semivalue'] != (1, 1):
            return 1
    else:
        if arg['paired_sampling']:
            return 1
    
    if arg['dataset_id'] in [40996, 1478]:
        if arg['n_queries_per_player'] != 100:
            return 1
        if arg['n_queries_per_player_per_checkpoint'] != 2:
            return 1
    elif arg['dataset_id'] == 40926:
        if arg['n_queries_per_player'] != 20:
            return 1
        if arg['n_queries_per_player_per_checkpoint'] != 2:
            return 1
    else:
        if arg['n_queries_per_player'] != 1000:
            return 1
        if arg['n_queries_per_player_per_checkpoint'] != 10:
            return 1
        
    
if __name__ == '__main__':
    from args import process_arg_dict
    import argparse
    from utilFuncs import treeUtility
    from createTreeModel import createTreeModel, _classification_ids
    from estimators import estimators
    
    keys_pop = ['path_estimate', 'path_groundtruth', 'dataset_id', 'n_estimators', 
                'random_seed_anythingelse']
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=1, help="number of processes")
    n_processes = parser.parse_args().p
    print('number of processes:', n_processes)
    
    args = process_arg_dict(arg_dict)
    n_total = len(args)
    
    for i, arg in enumerate(args):
        if skip_arg(arg):
            continue

        # create the utility function
        if not os.path.exists(arg['path_groundtruth']) or not os.path.exists(arg['path_estimate']):
            print(f'{i+1} / {n_total}')
            model, X_test, y_test = createTreeModel(arg['dataset_id'], 
                                                    arg['n_estimators'],
                                                    arg['random_seed_anythingelse'])
            np.random.seed(arg['random_seed_anythingelse'] + arg['dataset_id'])
            idx = np.random.choice(len(X_test))
            x = X_test[idx]
            if arg['dataset_id'] in _classification_ids:
                n_classes = len(np.unique(y_test))
                idx = np.random.choice(n_classes)
                util = treeUtility(model, x, idx)
            else:
                util = treeUtility(model, x)
            
        #compute the groundtruth
        if not os.path.exists(arg['path_groundtruth']):
            print(arg['path_groundtruth'])
            path_components = arg['path_groundtruth'].split(os.sep)
            os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
            groundtruth = util.treestab(arg['semivalue'])
            np.savez_compressed(arg['path_groundtruth'], groundtruth=groundtruth)
            
        if not os.path.exists(arg['path_estimate']):
            print(arg['path_estimate'])
            path_estimate = arg['path_estimate']
            path_components = path_estimate.split(os.sep)
            os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
            
            for key in keys_pop:
                arg.pop(key)
           
            est = estimators(arg.pop('n_queries_per_player'), 
                             arg.pop('n_queries_per_iteration'),
                             arg.pop('n_queries_per_player_per_checkpoint'), 
                             n_processes)
            estimate_traj = est.run(util, **arg)
            
            np.savez_compressed(path_estimate, estimate_traj=estimate_traj)