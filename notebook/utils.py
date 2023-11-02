import json
import os

from matplotlib import pyplot as plt
import numpy as np
import itertools


def get_atk_data(data_folder):
    
    file_names = []
    for df in data_folder:
        latest_file_index = max([int(f) for f in os.listdir(df) if not f.startswith('_') and not f.startswith('.')])
        file_names.append(os.path.join(df, str(latest_file_index), 'info.json'))

    data_list = []
    for f in file_names:
        f = open(f,)
        data_list.append(json.load(f))

    return_mean_list, return_std_list, adv_eps_list, eps_norms_info_list = [], [], [], []
    for data in data_list:
        test_return_mean = np.array([a['value'] for a in data['test_return_mean']])
        test_return_std = np.array([a['value'] for a in data['test_return_std']])
        
        adv_eps = np.array([eps for eps in data['test_adv_epsilon']])
        if 'test_epsilon_l1_norm' in data.keys():
            eps_l1norm = np.array([val for val in data['test_epsilon_l1_norm']])
            eps_l2norm = np.array([val for val in data['test_epsilon_l2_norm']])
            eps_linfnorm = np.array([val for val in data['test_epsilon_linf_norm']])
            eps_norms_info = {
                'l1_norm': eps_l1norm,
                'l2_norm': eps_l2norm,
                'linf_norm': eps_linfnorm,
            }
        else:
            eps_norms_info = None
        
        return_mean_list.append(test_return_mean)
        return_std_list.append(test_return_std)
        adv_eps_list.append(adv_eps)
        eps_norms_info_list.append(eps_norms_info)
    return return_mean_list, return_std_list, adv_eps_list, eps_norms_info_list

def plot_results_with_conf(return_mean_list, return_std_list, adv_eps_list, legend, xlab=None, 
                           ylab=None, title=None, alpha=0.05, xlim=None, 
                           linewidth=None, ylim=None, colors=None, linestyle=None, figsize=(15,12)):

    if colors is None:
        colors = ['C1','C2','C5','C3','b','C6','C7','C8','C9']
        
    if linestyle is None:
        linestyle = ['-']*len(colors)

    f, a = plt.subplots(figsize=figsize)
    
    if linewidth == None:
        linewidth = [3] * len(return_mean_list)
        
    pa_list, pb_list = [], []
    max_eps = float('-inf')
    for i, (return_mean, return_std, adv_eps, lb, lw) in enumerate(zip(return_mean_list, return_std_list, adv_eps_list, legend, linewidth)):
        pa = a.plot(adv_eps, return_mean, linestyle=linestyle[i], color=colors[i], linewidth=lw)

        a.fill_between(adv_eps, return_mean-return_std, return_mean+return_std,
            alpha=alpha, edgecolor=colors[i], facecolor=colors[i], label=lb)
        pb = a.fill(np.NaN, np.NaN, color=colors[i], alpha=0.2)
        
        pa_list.append(pa)
        pb_list.append(pb)
        max_eps = max(max_eps, np.max(adv_eps).item())
        
    ncol = 2 if len(legend) > 3 else len(legend)
    lgd = a.legend([(pa[0],pb[0]) for pa,pb in zip(pa_list,pb_list)], legend, 
                   loc='upper right')

    if xlab is not None:
        a.set_xlabel(xlab)
    if ylab is not None:
        a.set_ylabel(ylab)
    if title is not None:
        a.set_title(title,fontweight="bold")
    if xlim is not None:
        plt.xlim([0,xlim])
    else:
        plt.xlim([0,max_eps])

    if ylim is not None:
        plt.ylim(ylim)
        
    plt.xticks()
    plt.yticks()

    plt.show()
    return f, lgd