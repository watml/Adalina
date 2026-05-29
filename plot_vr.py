import os
import matplotlib.pyplot as plt
import numpy as np

path_groundtruth = os.path.join(
    'exp',
    'dataset_id={}',
    'n_estimators=10',
    'semivalue=(1, 1)',
    'groundtruth.npz'
    )

path_fig = os.path.join(
    'figs_vr',
    'dataset_id={}-vr.pdf'
    )


def plot_vr(x, errors, grand_util, empty_util, path_fig):
    path_components = path_fig.split(os.sep)
    os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(32, 24))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    curve_mean = errors.mean(axis=1)
    curve_std = errors.std(axis=1)
    ax.plot(x, curve_mean, linewidth=10)
    ax.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2)
    
    plt.axvline(x=grand_util, linestyle='--', label=r'$u_{[n]}$', color='tab:red', linewidth=10)
    plt.axvline(x=empty_util, linestyle='-.', label=r'$u_{\emptyset}$', color='tab:orange', linewidth=10)
        
    ax.tick_params(axis='x', labelsize=80)
    ax.tick_params(axis='y', labelsize=80)
    plt.legend(fontsize=100, framealpha=0.5)
    plt.xlabel(r'$\gamma$', fontsize=100)
    plt.ylabel(r'$\|\hat{\phi}^{\gamma}-\phi\|_{2} / \|\phi\|_{2}$', fontsize=100)
    
    plt.savefig(path_fig, bbox_inches='tight')
    plt.close(fig)



if __name__ == '__main__':
    from main_vr import arg_dict
    from args import dict2comb
    from collections import defaultdict
    from createTreeModel import createTreeModel
    from utilFuncs import treeUtility
    from matplotlib.ticker import FormatStrFormatter
    
    args = dict2comb(arg_dict)
    
    args_per_id = defaultdict(list)
    for arg in args:
        args_per_id[arg['dataset_id']].append(arg)
              
    for dataset_id, args_2nd in args_per_id.items():
        arg0 = args_2nd[0]
        model, X_test, y_test = createTreeModel(arg0['dataset_id'], 
                                                arg0['n_estimators'],
                                                arg0['random_seed_anythingelse'])
        np.random.seed(arg0['random_seed_anythingelse'] + arg0['dataset_id'])
        idx = np.random.choice(len(X_test))
        x = X_test[idx]
        n_classes = len(np.unique(y_test))
        idx = np.random.choice(n_classes)
        util = treeUtility(model, x, idx)

        pos = arg['position']
        n_players = len(x)
        grand_util = util.evaluate(np.ones(n_players))
        empty_util = util.evaluate(np.zeros(n_players))
        limit = np.max(np.abs([grand_util, empty_util])) * 1.5
        
        x = limit / 40 * np.array(arg_dict['position'], dtype=np.float64)

        data = np.load(path_groundtruth.format(arg0['dataset_id']))
        groundtruth = data['groundtruth']
        
        errors = np.empty((len(arg_dict['position']), len(arg_dict['random_seed_estimator'])), dtype=np.float64)
        
        for arg in args_2nd:
            arg['path'] = arg['path'].format(arg['dataset_id'],
                                             arg['semivalue'],
                                             arg['random_seed_estimator'],
                                             arg['position'])
            data = np.load(arg['path'])
            estimate = data['estimate_traj'][-1]
            errors[arg['position'] // 2 + 20, arg['random_seed_estimator']] = np.linalg.norm(estimate - groundtruth) / np.linalg.norm(groundtruth)
            
        plot_vr(x, errors, grand_util, empty_util, path_fig.format(arg['dataset_id']))
            