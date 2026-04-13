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
    path=os.path.join(
        'exp',
        'VR',
        'dataset_id={}',
        'semivalue={}',
        'random_seed_estimator={}',
        'pos={}.npz'
        ),
    n_queries_per_player=1000,
    n_queries_per_iteration=100,
    n_queries_per_player_per_checkpoint=10,
    random_seed_anythingelse=2026,
    n_estimators=10,
    # varied
    dataset_id=[44, 1475, 41145, 41150],
    semivalue=[(1, 1), 0.5],
    position= np.arange(-40, 41, 2),
    random_seed_estimator=np.arange(10),
    )



if __name__ == '__main__':
    import argparse
    from utilFuncs import treeUtility
    from createTreeModel import createTreeModel, _classification_ids
    from estimators import estimators
    from args import dict2comb
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", type=int, default=1, help="number of processes")
    n_processes = parser.parse_args().p
    print('number of processes:', n_processes)
    
    args = dict2comb(arg_dict)
    
    n_total = len(args)
    
    for i, arg in enumerate(args):
        arg['path'] = arg['path'].format(arg['dataset_id'],
                                         arg['semivalue'],
                                         arg['random_seed_estimator'],
                                         arg['position'])
        
        if not os.path.exists(arg['path']):
            print(f'{i+1} / {n_total}.')
            print(arg['path'])
            
            path = arg.pop('path')
            path_components = path.split(os.sep)
            os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
            
            model, X_test, y_test = createTreeModel(arg['dataset_id'], 
                                                    arg.pop('n_estimators'),
                                                    arg['random_seed_anythingelse'])
            np.random.seed(arg.pop('random_seed_anythingelse') + arg['dataset_id'])
            idx = np.random.choice(len(X_test))
            x = X_test[idx]
            if arg.pop('dataset_id') in _classification_ids:
                n_classes = len(np.unique(y_test))
                idx = np.random.choice(n_classes)
                util = treeUtility(model, x, idx)
            else:
                util = treeUtility(model, x)
                
            
            pos = arg.pop('position')
            n_players = len(x)
            grand_util = util.evaluate(np.ones(n_players))
            empty_util = util.evaluate(np.zeros(n_players))
            limit = np.max(np.abs([grand_util, empty_util])) * 1.5
            arg.update(aux = limit / 40 * pos)
                          
            est = estimators(arg.pop('n_queries_per_player'), 
                             arg.pop('n_queries_per_iteration'),
                             arg.pop('n_queries_per_player_per_checkpoint'), 
                             n_processes)
            
            estimate_traj = est.run(util, 'linearAppr', **arg)
            
            np.savez_compressed(path, estimate_traj=estimate_traj)
            
            
            
    
    
    
    
    
    
    