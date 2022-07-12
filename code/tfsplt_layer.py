import glob
import os
import argparse
from threading import Condition
import pandas as pd
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tfsplt_utils import read_sig_file, read_folder
from tfsenc_parser import parse_arguments
from utils import main_timer


# -----------------------------------------------------------------------------
# Utils functions
# -----------------------------------------------------------------------------

def get_child_folders(args):
    """Get child folders given parent folder and conditions

    Args:
        args (namespace): commandline arguments

    Returns:
        folders (Dict): condition (key): list of child folder paths (value)
    """  
    folders = {}
    for condition in args.conditions:
        format = glob.glob(args.top_dir + f'/*-{condition}-*') # all children with the formats
        if condition == 'all':
            format = [ind_format for ind_format in format if 'flip' not in ind_format] # blenderbot
        if args.is_test:
            format = [ind_format for ind_format in format if '47' in ind_format or '48' in ind_format] # for testing
        folders[condition] = format
    return folders


def get_sigelecs(args):
    """Get significant electrodes

    Args:
        args (namespace): commandline arguments

    Returns:
        sigelecs (Dict): mode (key): list of sig elecs (value)
    """  
    # Sig Elecs
    sigelecs = {}
    if args.sig_elecs:
        for mode in args.modes:
            sig_file_name = f'tfs-sig-file-{str(args.sid)}-sig-1.0-{mode}.csv'
            if args.sid == 777:
                sig_file_name = 'podcast_160.csv'
            sigelecs[mode] = read_sig_file(sig_file_name)
    return sigelecs


def get_layer_ctx(args, dir_path):
    """Get the layer number and context length of a child folder

    Args:
        args (namespace): commandline arguments
        dir_path (string): child folder path

    Returns:
        layer (int): layer number
        ctx_len (int|None): context length
    """  
    # get layer number from directory path
    layer_idx = dir_path.rfind('-')
    layer = dir_path[(layer_idx + 1):]
    assert layer.isdigit(), "Need layer to be an int"
    layer = int(layer)

    # get context length from directory path
    if args.has_ctx:
        partial_format = dir_path[:layer_idx]
        ctx_len = partial_format[(partial_format.rfind('-') + 1):]
        assert ctx_len.isdigit(), 'Need context length to be an int'
        ctx_len = int(ctx_len) 
    else:
        ctx_len = None
    return (layer, ctx_len)


# -----------------------------------------------------------------------------
# Argument Parser
# -----------------------------------------------------------------------------

def parse_arguments():
    """Argument parser

    Args:

    Returns:
        args (namespace): commandline arguments
    """  
    parser = argparse.ArgumentParser()

    parser.add_argument("--sid", type=int, required=True)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--top-dir", type=str, required=True)
    parser.add_argument("--modes", nargs="+",  required=True)
    parser.add_argument("--conditions", nargs="+",  required=True)

    parser.add_argument('--has-ctx', action='store_true', default=False)
    parser.add_argument('--sig-elecs', action='store_true', default=False)

    parser.add_argument("--outfile", default='results/figures/ericplots.pdf')

    args = parser.parse_args()

    return args


def set_up_environ(args):
    """Set up some indirect arguments

    Args:
        args (namespace): commandline arguments

    Returns:
        args (namespace): commandline arguments
    """
    args.layers = np.arange(1, args.layer_num+1)
    args.sigelecs = get_sigelecs(args)
    args.lags =  np.arange(-10000, 10001, 25)
    args.is_test = False

    return args


# -----------------------------------------------------------------------------
# Aggregate Data Functions
# -----------------------------------------------------------------------------

def average_folder(dir_path, args, mode, condition):
    """Aggregate and average data for a specific child folder

    Args:
        dir_path: child folder path
        args (namespace): commandline arguments
        mode: comp or prod
        condition: condition for the child folder

    Returns:
        subdata_ave (dataframe): average correlation for the child folder
    """
    layer, ctx_len = get_layer_ctx(args, dir_path)
    if ctx_len < 16:
        return None
    print('cond:', condition, ' mode:', mode, ' context:', ctx_len, ' layer:', layer)

    subdata = []
    fname = dir_path + f'/*/*_{mode}.csv'
    subdata = read_folder(subdata, fname, args.sigelecs, mode)
    subdata = pd.concat(subdata)
    subdata_ave = subdata.describe().loc[['mean'],]
    subdata_ave = subdata_ave.assign(layer=layer,mode=mode,condition=condition)
    if args.has_ctx:
        subdata_ave = subdata_ave.assign(ctx = ctx_len)
    return subdata_ave


def aggregate_folders_parallel(args, folders, parallel = True):
    """Aggregate data for all folders

    Args:
        args (namespace): commandline arguments
        folders (Dict): dictionary of all child folders
        parallel (Bool): if aggregate data using pool

    Returns:
        df (dataframe): dataframe of all folders (one line per folder)
    """
    data = []
    p = Pool(10)

    for condition in args.conditions:
        for mode in args.modes:
            if parallel:
                print('Aggregating Data Parallel')
                for result in p.map(partial(average_folder, 
                        args=args,
                        mode=mode, 
                        condition=condition,
                    ), folders[condition]):
                    data.append(result)
            else:
                print('Aggregating Data')
                for child_folder in folders[condition]:
                    data.append(average_folder(child_folder,args,mode,condition))

    print('Organizing Data')
    df = pd.concat(data)
    # if not args.is_test:
    #     assert len(df) == args.layer_num * len(args.conditions) * len(args.modes)
    if args.has_ctx:
        df = df.sort_values(by=['condition','mode','ctx','layer'])
        df.set_index(['condition','mode','ctx','layer'], inplace = True)
    else:
        df = df.sort_values(by=['condition','mode','layer'])
        df.set_index(['condition','mode','layer'], inplace = True)
    
    return df


# -----------------------------------------------------------------------------
# Plotting Functions
# -----------------------------------------------------------------------------

def ericplot1(args, df, pdf):

    cmap = plt.cm.get_cmap('jet')
    marker_diff = 0.01

    fig, axes = plt.subplots(len(args.conditions),len(args.modes),figsize=(18,5*len(args.conditions)))
    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i,j]
            encrs = df.loc[(condition, mode), :]
            max_lags = encrs.idxmax(axis=1)
            yheights = np.ones(encrs.shape[-1]) * 1.01 + marker_diff
            encrs = encrs.divide(encrs.max(axis=1),axis=0)
            for layer in args.layers:
                ax.plot(args.lags, encrs.loc[layer], c=cmap(layer/args.layer_num), zorder=-1, lw=0.5)
                maxlag = max_lags[layer]
                ax.scatter(args.lags[maxlag], yheights[maxlag], marker='o', color=cmap(layer/args.layer_num))
                yheights[maxlag] += marker_diff
            ax.set(ylim=(0.8, 1.2), xlim=(-2000, 1000))
            if i == 0:
                ax.title.set_text(mode)
            if j == 0:
                ax.set_ylabel(condition)
            if j == 1:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='4%', pad=0.05)
                sm = plt.cm.ScalarMappable(cmap=cmap)
                sm.set_array([])
                cbar = fig.colorbar(sm, cax = cax,orientation="vertical")
                # cbar = fig.colorbar(sm, ax=ax)
                cbar.set_ticks([0,0.25, 0.5, 0.75,1])
                cbar.set_ticklabels([1, int(args.layer_num/4), int(args.layer_num/2),int(3*args.layer_num/4), args.layer_num])

    pdf.savefig(fig)
    plt.close()
    return pdf


def ericplot2(args, df, pdf):
    colors = ('blue','red','green','black')
    fig, axes = plt.subplots(len(args.conditions),len(args.modes), figsize=(18,5*len(args.conditions)))
    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i,j]
            encrs = df.loc[(condition, mode), :]
            max_time = args.lags[encrs.idxmax(axis=1)]  # find the lag for the maximum 
            ax.scatter(args.layers, max_time, c=colors[j])
            # Fit line through scatter
            linefit = stats.linregress(args.layers, max_time)
            ax.plot(args.layers, args.layers*linefit.slope + linefit.intercept, ls='--', c=colors[j])
            ax.set(ylim=(-2000,1000))
            ax.text(3, ax.get_ylim()[1]-200, f'r={linefit.rvalue:.2f} p={linefit.pvalue:.2f}')
            if i == 0:
                ax.set_title(mode)
            if j == 0:
                ax.set_ylabel(condition)
    pdf.savefig(fig)
    plt.close()
    return pdf


def ericplot3(args, df, pdf, type):
    ticks = [1, args.layer_num/2, args.layer_num]
    fig, axes = plt.subplots(len(args.conditions),len(args.modes), figsize=(18,5*len(args.conditions)))

    for i, condition in enumerate(args.conditions):
        for j, mode in enumerate(args.modes):
            ax = axes[i,j]

            if type == 'mean(2s)':
                chosen_lag_idx = [idx for idx, element in enumerate(args.lags) if element <=2000 and element >= -2000]
                encrs = df.loc[(condition, mode), chosen_lag_idx]
                mean = encrs.mean(axis=1)
                errs = encrs.std(axis=1)**2

                ax.bar(args.layers,mean,yerr=errs,align='center',alpha=0.5,ecolor='black',capsize=0)
            elif type == 'max':
                encrs = df.loc[(condition, mode),:]
                max = encrs.max(axis=1)
                ax.scatter(args.layers,max)

            ax.set(xlim=(0.5, args.layer_num+0.5), xticks=ticks, xticklabels=ticks)
            if i == 0:
                ax.set_title(f'{mode}-{type}')
            if j == 0:
                ax.set_ylabel(condition)

    pdf.savefig(fig)
    plt.close()
    return pdf


def get_idx_val(idx, start, gap):
    return start + idx * gap


def heatmap(df, pdf, type):
    # sns.dark_palette("#69d", reverse=True, as_cmap=True)
    if type == 'max':
        plot_mat = df.max(axis=1).unstack([0,1,2]).sort_values(by='layer',ascending=False)
    elif type == 'idxmax':
        plot_mat = df.idxmax(axis=1).unstack([0,1,2])
        start_lag = -500 # -2000 or -500
        gap_lag = 5 # 25 or 5
        plot_mat = plot_mat.apply(get_idx_val, args=(start_lag,gap_lag))
    fig, ax = plt.subplots(figsize=(15,6))
    ax = sns.heatmap(plot_mat, cmap='Blues')
    pdf.savefig(fig)
    plt.close()
    return pdf


@main_timer
def main():

    args = parse_arguments()
    args = set_up_environ(args)

    args.is_test = True
    folders = get_child_folders(args) # get child folders
    df = aggregate_folders_parallel(args, folders) # aggregate data

    print('Plotting')
    pdf = PdfPages(args.outfile)

    pdf = ericplot1(args, df, pdf)
    pdf = ericplot2(args, df, pdf)
    pdf = ericplot3(args, df, pdf, 'mean(2s)')
    pdf = ericplot3(args, df, pdf, 'max')

    # pdf = heatmap(df, pdf, 'max')
    # pdf = heatmap(df, pdf, 'idxmax')

    pdf.close()

    return


if __name__ == "__main__":
    main()
