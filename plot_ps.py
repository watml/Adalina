import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def skip_arg(arg):
    if arg['n_estimators'] == 10:
        return 1
    
    if arg['n_queries_per_player'] != 1000:
        return 1
    
    if arg['n_queries_per_player_per_checkpoint'] != 10:
        return 1
    
    if arg['dataset_id'] not in [44, 1475, 41145, 41150]:
        return 1
    
    if arg['estimator'] != 'kernelSHAP_MV':
        return 1
    
    if arg['semivalue'] != (1, 1):
        return 1
    return 0

def plot_curves(curves_all, dataset_id):
    path = 'figs_ps'
    os.makedirs(path, exist_ok=True)
    
    palette = sns.color_palette("tab10")
    x = np.arange(10, 1001, 10)
     
    for j in [0, 1]:
        fig, ax = plt.subplots(figsize=(32, 24))
            
        for i, (key, curves) in enumerate(curves_all.items()):  
            
            curve_mean = curves[j, 0].mean(axis=0)
            curve_std = curves[j, 0].std(axis=0)   
            ax.plot(x, curve_mean, linewidth=10, label=key, c=palette[i])
            ax.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=palette[i])
            
            curve_mean = curves[j, 1].mean(axis=0)
            curve_std = curves[j, 1].std(axis=0)   
            ax.plot(x, curve_mean, linewidth=10, linestyle='--', c=palette[i])
            ax.fill_between(x, curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=palette[i])
            
    
        ax.tick_params(axis='x', labelsize=80, pad=15)
        ax.tick_params(axis='y', labelsize=80)
        plt.yscale('log')
        plt.xscale('log')
        plt.ylabel(r'$\|\hat{\phi}-\phi\|_{2} / \|\phi\|_{2}$', fontsize=100)
        plt.xlabel('#utility queries per player', fontsize=100)
        
        plt.legend(fontsize=100)
        if j:
            path_fig = os.path.join(path, f'dataset_id={dataset_id}-both.pdf')
        else:
            path_fig = os.path.join(path, f'dataset_id={dataset_id}-positive.pdf')
        plt.savefig(path_fig, bbox_inches='tight')
        plt.close(fig)
        


if __name__ == '__main__':
    from main import arg_dict
    from args import process_arg_dict
    from collections import defaultdict
    
    args = process_arg_dict(arg_dict)
    
    args_per_id = defaultdict(list)
    for arg in args:
        if skip_arg(arg):
            continue
        args_per_id[arg['dataset_id']].append(arg)
        
    for dataset_id, args_2nd in args_per_id.items():
        curves_all = defaultdict(lambda : np.empty((2, 2, 10, 100), dtype=np.float64))
        for arg in args_2nd:
            groundtruth = np.load(arg['path_groundtruth'])['groundtruth']
            estimate = np.load(arg['path_estimate'])['estimate_traj']
            error = np.linalg.norm(estimate - groundtruth[None, :], axis=1) / np.linalg.norm(groundtruth)
            curves_all[arg['sampling']][0 if arg['n_estimators']==0 else 1, arg['paired_sampling'], arg['random_seed_estimator']] = error
            
        plot_curves(curves_all, dataset_id)