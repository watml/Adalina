import itertools
import numpy as np
from estimators import can_estimator_run_semivalue, can_estimator_be_paired
import os
from collections import defaultdict


path_structure = [
    'dataset_id',
    'n_estimators',
    'semivalue',
    'random_seed_estimator',
    'estimator  aux  scalar_vr  sampling  unbiased  paired_sampling',
    ]



def process_arg_dict(arg_dict):
    arg_dict_flatten = flatten_arg_dict(arg_dict)
    arg_comb = dict2comb(arg_dict_flatten)
    arg_comb = filter_arg(arg_comb)
    add_path(arg_comb)
    arg_comb = sort(arg_comb, arg_dict['dataset_id'])
    
    return arg_comb


def flatten_arg_dict(arg_dict):
    unprocessed = [arg_dict]
    arg_dict_flatten = []
    while len(unprocessed):
        dict_cur = unprocessed.pop()
        flag = True
        for key, value in dict_cur.items():
            if isinstance(value, dict):
                dict_cur.pop(key)
                for name, component in value.items():
                    dict_new = dict_cur.copy()
                    if isinstance(component, list):
                        dict_new[key] = component
                    elif isinstance(component, dict):
                        dict_new[key] = name
                        dict_new.update(component)
                    else:
                        raise ValueError
                    unprocessed.append(dict_new)
                flag = False
                break
        if flag:
            arg_dict_flatten.append(dict_cur) 
    return arg_dict_flatten
            

def dict2comb(arg_dict_flatten):
    if not isinstance(arg_dict_flatten, list):
        assert isinstance(arg_dict_flatten, dict)
        arg_dict_flatten = [arg_dict_flatten]
    
    arg_comb = []
    for dict_cur in arg_dict_flatten:
        for key, value in dict_cur.items():
            if isinstance(value, np.ndarray):
                dict_cur[key] = value.tolist()
            elif isinstance(value, range):
                dict_cur[key] = list(value)
            elif not isinstance(value, list):
                dict_cur[key] = [value]     
                
        keys = dict_cur.keys()
        values = dict_cur.values()
        for instance in itertools.product(*values):
            arg_comb.append(dict(zip(keys, instance)))
           
    return arg_comb


def filter_arg(arg_comb):
    arg_comb_filtered = []
    for arg in arg_comb:
        if not can_estimator_run_semivalue(arg['estimator'], arg['semivalue']):
            continue
        
        if arg['paired_sampling']:
            if not can_estimator_be_paired(arg['estimator'], arg['semivalue']):
                continue
        
        arg_comb_filtered.append(arg)
    return arg_comb_filtered
        

def add_path(arg_comb):
    structure = []
    for item in path_structure:
        structure.append(item.split())
    
    for arg in arg_comb:
        path = arg.pop('root')
        for keys in structure:
            path_cur = ''
            for key in keys:
                if key in arg.keys():
                    path_cur += f'{key}={arg[key]}-'
            path_cur = path_cur[:-1]
            path = os.path.join(path, path_cur)  
        arg['path_estimate'] = path + '.npz'
        
        path_components = arg['path_estimate'].split(os.sep)
        path_groundtruth = os.sep.join(path_components[:4])
        arg['path_groundtruth'] = os.path.join(path_groundtruth, 'groundtruth.npz')
        
        
def sort(arg_comb, dataset_ids):
    arg_sorted = defaultdict(list)
    for arg in arg_comb:
        arg_sorted[arg['dataset_id']].append(arg)
    arg_comb = []
    for dataset_id in dataset_ids:
        arg_comb += arg_sorted[dataset_id]
        
    return arg_comb
    
    
    
                        