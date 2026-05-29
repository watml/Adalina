import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


path_fig = os.path.join(
    'figs_adaptive',
    'dataset_id={}-{}-adaptive.pdf'
    )

weighted_banzhaf =  np.arange(.1, 1, .1).round(1).tolist()
beta_shapley = [(16, 1), (4, 1), (1, 1), (1, 4), (1, 16), (16, 4), (2, 2), (8, 8), (4, 16)]


def skip_arg(arg):
    if arg['n_queries_per_player'] != 1000:
        return 1
    
    if arg['n_queries_per_player_per_checkpoint'] != 10:
        return 1
    
    if arg['n_estimators'] == 0:
        return 1
    
    if arg['estimator'] not in ['linearAppr', 'Adalina', 'SHAP_IQ']:
        return 1   
    
    if arg['paired_sampling']:
        return 1
    
    if arg['dataset_id'] not in [44, 1475, 41145, 41150]:
        return 1

    return 0

def plot_curves(results, dataset_id):
    est2key = {
        'linearAppr-empty' : r'$\gamma=u_{\emptyset}$',
        'linearAppr-default' : r'$\gamma=\frac{u_{[n]}+u_{\emptyset}}{2}$',
        'Adalina' : 'adaptive',
        'SHAP_IQ' : 'SHAP-IQ'
        }
    ests = ['linearAppr-empty', 'linearAppr-default', 'Adalina', 'SHAP_IQ']

    
    for j in [0, 1]:  
        if j:
            path = path_fig.format(dataset_id, 'banzhaf')
        else:
            path = path_fig.format(dataset_id, 'shapley')
            
        path_components = path.split(os.sep)
        os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(32, 24))
        palette = sns.color_palette("tab10")
        palette = [palette[1], palette[3], palette[0], palette[2]]
        
        curves_all = []
        for est in ests:
            curves_all.append(results[est][j])
        
        
        for i, (key, curves) in enumerate(zip(ests, curves_all)):
            key = est2key[key]
                     
            curve_mean = curves.mean(axis=1)
            curve_std = curves.std(axis=1)
            
            ax.semilogy(np.arange(9), curve_mean, linewidth=10, label=key, c=palette[i])
            ax.fill_between(np.arange(9), curve_mean - curve_std, curve_mean + curve_std, alpha=0.2, color=palette[i])
            
        ax.tick_params(axis='x', labelsize=80)
        ax.tick_params(axis='y', labelsize=80)
        
        if j:
            plt.xlabel('Weighted Banzhaf values', fontsize=100)
            plt.xticks(np.arange(9), [str(e) for e in weighted_banzhaf])
        else:
            plt.xlabel('Beta Shapley values', fontsize=100)
            plt.xticks(np.arange(9), [str(e) for e in beta_shapley])
            xticklabels = ax.get_xticklabels()
            for i in range(9):
                xticklabels[i].set_rotation(-90)

        
        plt.legend(fontsize=100)
        plt.ylabel(r'$\|\hat{\phi}-\phi\|_{2} / \|\phi\|_{2}$', fontsize=100)
        plt.savefig(path, bbox_inches='tight')
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
        results = defaultdict(lambda : np.empty((2, 9, 10), dtype=np.float64))
        for arg in args_2nd:
            est = arg['estimator']
            if 'aux' in arg:
                est += '-' + arg['aux']
                
            data = np.load(arg['path_groundtruth'])
            groundtruth = data['groundtruth']          
            data = np.load(arg['path_estimate'])
            estimate = data['estimate_traj'][-1]          
            error = np.linalg.norm(estimate - groundtruth) / np.linalg.norm(groundtruth)
            
            if isinstance(arg['semivalue'], tuple):
                results[est][0, beta_shapley.index(arg['semivalue']), arg['random_seed_estimator']] = error
            else:
                results[est][1, weighted_banzhaf.index(arg['semivalue']), arg['random_seed_estimator']] = error
                
                
                
        plot_curves(results, dataset_id)
                
            
            

        