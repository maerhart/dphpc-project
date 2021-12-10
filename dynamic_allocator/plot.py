#!/usr/bin/env python
# coding: utf-8

# Plot benchmarks using existing run data

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sys


CSV_PATH = sys.argv[1]

header = ['repetition', 'version', 'workload', 'blocks', 'threads_per_block', 'floats', 'malloc_mean', 'malloc_max', 'free_mean', 'free_max', 'work_mean', 'work_max']
df = pd.read_csv(CSV_PATH+"/input.csv", names = header, delimiter = ' ')


# single plot
def plot_floats_vs_cycles(df, ax, params):
    
    measurement_type = params['measurement_type'] # is either "Malloc", "Work", or "Free"
    
    # filter for data that fits the given parameters
    df = df[df['blocks'] == params['blocks']\
          & df['threads_per_block'] == params['threads_per_block']\
          & df['workload'] == params['workload']]
    
    # calculate mean and std over all runs
    df = df.groupby(['floats', 'version'], as_index=False)\
           .agg({'malloc_mean':['mean','std'], 'malloc_max':['mean', 'std'],\
                 'free_mean':['mean','std'],   'free_max':['mean', 'std'],\
                 'work_mean':['mean','std'],   'work_max':['mean', 'std']}\
               )
    # set the new headers
    df.columns = ['floats', 'version', 'malloc_mean_mean', 'malloc_mean_std', 'malloc_max_mean', 'malloc_max_std',\
                                       'free_mean_mean', 'free_mean_std', 'free_max_mean', 'free_max_std',\
                                       'work_mean_mean', 'work_mean_std', 'work_max_mean', 'work_max_std'\
                 ]
    
    
    floats = list(df['floats'].unique())
    
    if 'Malloc' == params['measurement_type']:
        ax.set_title("Malloc")
        
        for version in list(df['version'].unique()):
            mean, std = df[df['version'] == version]['malloc_mean_mean', 'malloc_mean_std']
            ax.errorbar(floats, mean, std, label=str(version)+": Malloc time")
            
    elif 'Free' == params['measurement_type']:
        ax.set_title("Free")
        
        for version in list(df['version'].unique()):
            mean, std = df[df['version'] == version]['free_mean_mean', 'free_mean_std']
            ax.errorbar(floats, mean, std, label=str(version)+": Free time")
        
    elif 'Work' == params['measurement_type']:
        ax.set_title("Work")
        
        for version in list(df['version'].unique()):
            mean, std = df[df['version'] == version]['work_mean_mean', 'work_mean_std']
            ax.errorbar(floats, mean, std, label=str(version)+": Work time")
        
    
    ax.legend()
    ax.set_xlabel("#floats")
    ax.set_ylabel("#clock cycles")
    
    
    
# generate a 1x3 plot
fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 30))

for i, measurement_type in enumerate(['Malloc', 'Work', 'Free']):
    params = {'measurement_type': measurement_type,
              'blocks': 64,
              'threads_per_block': 64,
              'workload': 'sum_all'}
    plot_floats_vs_cycles(df, axs[i], params)
    
plt.savefig(CSV_PATH+"/result.png", dpi=60, bbox_inches='tight')   
plt.show()


def plot(c, nc, x, x_name, y, y_name, title):
    font = {'weight' : 'bold',
            'size'   : 22}
    matplotlib.rc('font', **font)
    fontdict = {'fontsize': 28,
                'fontweight': 'bold'}
    fig, axs = plt.subplots(len(y), len(x), sharey=True, sharex=True, figsize=(40, 40))
    for j, val_x in enumerate(x):
        for i, val_y in enumerate(y):
            axs[i, j].grid(True)
            axs[i, j].set_yscale('log', base=2)
        axs[0, j].set_title(x_name + ": " + str(val_x), fontdict=fontdict)
    
    for ax, row in zip(axs[:,0], y):
        ax.set_ylabel(y_name + ": " + str(row), fontdict=fontdict)
    
    for j, val_x in enumerate(x):
        for i, val_y in enumerate(y):
            cred = c.loc[(c[x_name] == val_x) & (c[y_name] == val_y)]
            ncred = nc.loc[(nc[x_name] == val_x) & (nc[y_name] == val_y)]
            axs[i, j].errorbar(list(c['ints'].unique()), cred['time_mean'], yerr=cred['time_std'], label='Coalesced', ls='-', lw=4, ms=20, c='orange')
            axs[i, j].errorbar(list(c['ints'].unique()), ncred['time_mean'], yerr=ncred['time_std'], label='Non Coalesced', ls='-', lw=4, ms=20, c='blue')
            axs[i, j].get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            ax.set_ylabel(y_name + ": " + str(row), fontdict=fontdict)
            num_col = val_y
            if not BY_BLOCK:
                num_col = num_col * val_x
            elif x_name == 'num_blocks':
                num_col = val_x
            ax2 = axs[i, j].twinx()
            #ax2.set_yscale('log', base=2)
            ax2.plot(list(c['ints'].unique()), cred['alloc_failures_mean']/num_col, label='Coalesced Alloc Failures', ls=':', lw=4, ms=20, c='orange')
            ax2.plot(list(c['ints'].unique()), ncred['alloc_failures_mean']/(val_x*val_y), label='Non Coalesced Alloc Failures', ls=':', lw=4, ms=20, c='blue')
            if(j == len(x)-1):
                ax2.set_ylabel('Number of Failures', fontdict=fontdict)
            else:
                ax2.set_yticks([])
            
            
    fig.text(0.0, 0.5, 'Time taken [msec]', ha='center', va='center', rotation='vertical', fontsize=42)
    fig.text(0.5, 0.0, 'Number of ints allocated (power of 2)', ha='center', va='center', fontsize=42)
    for j in range(len(x)):
        for i in range(len(y)):
            axs[i, j].legend(loc="upper left")
    fig.align_ylabels()
    fig.tight_layout()
    plt.savefig(title+".png", dpi=60, bbox_inches='tight')
    plt.show()
    
    
plot(c, nc, threads, 'num_threads', blocks, 'num_blocks', CSV_PATH+'/results') 