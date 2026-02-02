import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


estimators = {
    'Adalina_All' : 'Adalina-All (ours)',
    'Adalina' : 'Adalina (ours)', 
    'MSR_Wang' : 'MSR-Wang',
    'MSR_Witter' : 'MSR-Witter',
    'SHAP_IQ' : 'SHAP-IQ',
    'kernelSHAP_MV' : 'kernelSHAP',
    'linearAME' : 'AME',
    'ARM' : 'ARM',
    'GELS' : 'GELS',
    'GELS_Shapley' : 'GELS-Shapley'
    }

path_fig = os.path.join(
    'fig_comparison',
    'dataset_id={}-{};comparison.pdf'
    )

est_shapley = ['GELS_Shapley', 'kernelSHAP_MV']


p = sns.color_palette("tab10")
palette = [p[3], p[0], p[4], p[-1]] + sns.color_palette('Set2')

weighted_banzhaf =  np.arange(.1, 1, .1).round(1).tolist()
beta_shapley = [(16, 1), (4, 1), (1, 1), (1, 4), (1, 16), (16, 4), (2, 2), (8, 8), (4, 16)]


def skip_arg(arg):
    if arg['estimator'] not in estimators:
        return 1
    if arg['paired_sampling']:
        return 1
    if arg['estimator'] == 'kernelSHAP_MV':
        if arg['scalar_vr'] != 'diff' or arg['sampling'] != 'geomean':
            return 1
    return 0


def plot_curves(results, dataset_id):   
    for j in [0, 1]:  
        if j:
            path = path_fig.format(dataset_id, 'banzhaf')
        else:
            path = path_fig.format(dataset_id, 'shapley')
            
        path_components = path.split(os.sep)
        os.makedirs(os.sep.join(path_components[:-1]), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(32, 24))
        
        curves_all = []
        for est in estimators.keys():
            curves_all.append(results[est][j])
        
        
        for i, (est, curves) in enumerate(zip(estimators.keys(), curves_all)):
            key = estimators[est]
            
            if est == 'MSR_Wang' and j == 0:
                continue
            if est in est_shapley and j == 1:
                continue
            
            if est in est_shapley:
                pos = beta_shapley.index((1, 1))
                curve = curves[pos]
                plt.errorbar(pos, curve.mean(), yerr=curve.std(), fmt='o', markersize=30, color=palette[i],
                             ecolor=palette[i], capsize=18, elinewidth=5, capthick=5)

            else:                   
                curve_mean = curves.mean(axis=1)
                curve_std = curves.std(axis=1)
                if dataset_id == 41145 and est == 'linearAME' and j == 0:
                    # exclude one significant outlier
                    r = np.delete(curves[1], 5)
                    curve_mean[1] = r.mean()
                    curve_std = r.std()
                    
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

        
        plt.ylabel(r'$\|\hat{\phi}-\phi\|_{2} / \|\phi\|_{2}$', fontsize=100)
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        
    
def export_legend(legend, fig_saved):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(fig_saved, dpi="figure", bbox_inches=bbox)


if __name__ == '__main__':
    from main import arg_dict
    from args import process_arg_dict
    from collections import defaultdict
    
    for i, est in enumerate(estimators.keys()):
        plt.plot([], [], label=estimators[est], color=palette[i], linewidth=30)       
    legend = plt.legend(ncol=5, fontsize=100, loc="upper left", bbox_to_anchor=(1, 1))
    export_legend(legend, 'legend.pdf')
    
    
    args = process_arg_dict(arg_dict)
    
    args_per_id = defaultdict(list)
    for arg in args:
        if skip_arg(arg):
            continue
        args_per_id[arg['dataset_id']].append(arg)
        
    for dataset_id, args_2nd in args_per_id.items():
        results = defaultdict(lambda : np.zeros((2, 9, 10), dtype=np.float64))
        for arg in args_2nd:
            est = arg['estimator']
                
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
                
            
            

        