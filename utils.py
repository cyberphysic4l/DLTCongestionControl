from core.global_params import *
from matplotlib.lines import Line2D
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import plotly.graph_objects as go


def plot_cdf(data, xlabel: str, dirstr: str,  xlim=0):

    if SCHEDULING=='manaburn':
        mode_names = ['No Burn', 'Anxious', 'Greedy', 'Randomised Greedy']
        modes = list(set(BURN_POLICY))
    else:
        mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
        modes = list(set(MODE))    
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    _, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    step = STEP/10
    maxval = 0
    for NodeID in range(NUM_NODES):
        val = np.max(data[NodeID][0])
        if val>maxval:
            maxval = val
    maxval = max(maxval, xlim)
    Lines = [[] for _ in range(NUM_NODES)]
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval*1/step), 1)*step
            pdf = np.zeros(len(bins))
            i = 0
            if not isinstance(data[NodeID][0], int):
                if data[NodeID][0].size>1:
                    lats = sorted(data[NodeID][0])
                    for lat in lats:
                        while i<len(bins):
                            if lat>bins[i]:
                                i += 1
                            else:
                                break
                        pdf[i-1] += 1
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            if IOT[NodeID]:
                marker = 'x'
            else:
                marker = None
            if SCHEDULING=='manaburn':
                Lines[NodeID] = ax.plot(bins, cdf, color=colors[BURN_POLICY[NodeID]], linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
            else:
                Lines[NodeID] = ax.plot(bins, cdf, color=colors[MODE[NodeID]], linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        ax.legend(ModeLines, [mode_names[i] for i in modes], loc='lower right')
    plt.savefig(dirstr, bbox_inches='tight')
    
def plot_cdf_exp(data, ax):
    step = STEP/10
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(data[NodeID][0])>0:
            val = np.max(data[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    for NodeID in range(len(data)):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval*1/step), 1)*step
            pdf = np.zeros(len(bins))
            i = 0
            lats = sorted(data[NodeID][0])
            for lat in lats:
                while i<len(bins):
                    if lat>bins[i]:
                        i += 1
                    else:
                        break
                pdf[i-1] += 1
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            ax.plot(bins, cdf, color='tab:red')
    lmd = np.mean(data[1][0])
    ax.axvline(lmd, linestyle='--', color='tab:red')
    
    ax.set_title('rho = ' + str(1/(lmd*NU)))
    ax.plot(bins, np.ones(len(bins))-np.exp(-(1/lmd)*bins), color='black')
    #ax.plot(bins, np.ones(len(bins))-np.exp(-0.95*NU*bins), linestyle='--', color='tab:red')
    ModeLines = [Line2D([0],[0],color='tab:red', lw=2), Line2D([0],[0],linestyle='--',color='black', lw=2)]
    ax.legend(ModeLines, ['Measured',r'$1-e^{-\lambda t}$'], loc='lower right')

def per_node_barplot(data, xlabel: str, ylabel: str, title: str, dirstr: str, legend_loc: str = 'upper right', modes=None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.title.set_text(title)
    ax.set_ylabel(ylabel)
    if SCHEDULING=='manaburn':
        mode_names = ['No Burn', 'Anxious', 'Greedy', 'Randomised Greedy']
        modes = list(set(BURN_POLICY))
    else:
        mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
        modes = list(set(MODE))    
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if SCHEDULING=='manaburn':
            ax.bar(NodeID, data[NodeID], color=colors[BURN_POLICY[NodeID]])
        else:
            ax.bar(NodeID, data[NodeID], color=colors[MODE[NodeID]])
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)
    plt.savefig(dirstr, bbox_inches='tight')

def per_node_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window: int = 100, legend_loc: str = 'upper right', step=STEP, figtxt = ''):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.title.set_text(title)

    if SCHEDULING=='manaburn':
        mode_names = ['No Burn', 'Anxious', 'Greedy', 'Randomised Greedy']
        modes = list(set(BURN_POLICY))
    else:
        mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
        modes = list(set(MODE))    
    if SCHEDULING=='manaburn':
        mode_names = ['No Burn', 'Anxious', 'Greedy', 'Randomised Greedy']
    else:
        mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if np.any(data[:, NodeID]):
            if SCHEDULING=='manaburn':
                ax.plot(np.arange((avg_window-1)*step, SIM_TIME, step), np.convolve(np.ones(avg_window)/avg_window, data[:,NodeID], 'valid'), color=colors[BURN_POLICY[NodeID]], linewidth=5*REP[NodeID]/REP[0])
            else:
                ax.plot(np.arange((avg_window-1)*step, SIM_TIME, step), np.convolve(np.ones(avg_window)/avg_window, data[:,NodeID], 'valid'), color=colors[MODE[NodeID]], linewidth=5*REP[NodeID]/REP[0])
    
    ax.set_xlim(0, SIM_TIME)
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)

    plt.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(dirstr+'/plots/'+ylabel+'.png', bbox_inches='tight')

def scaled_rate_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window: int = 1000, legend_loc: str = 'right', modes = None, step=STEP, figtxt = ''):
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(8,8))
    ax[0].grid(linestyle='--')
    ax[1].grid(linestyle='--')
    ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[1].set_ylabel('Scaled '+ylabel)

    if SCHEDULING=='manaburn':
        mode_names = ['No Burn', 'Anxious', 'Greedy', 'Randomised Greedy']
        modes = list(set(BURN_POLICY))
    else:
        mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
        modes = list(set(MODE))    
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if MODE[NodeID] in modes and np.any(data[:, NodeID]):
            if SCHEDULING=='manaburn':
                ax[0].plot(np.arange(avg_window*STEP, SIM_TIME, STEP), (data[avg_window:,NodeID]-data[:-avg_window,NodeID])/(avg_window*STEP), linewidth=5*REP[NodeID]/REP[0], color=colors[BURN_POLICY[NodeID]])
                ax[1].plot(np.arange(avg_window*STEP, SIM_TIME, STEP), (data[avg_window:,NodeID]-data[:-avg_window,NodeID])*sum(REP)/(NU*REP[NodeID]*avg_window*STEP), linewidth=5*REP[NodeID]/REP[0], color=colors[BURN_POLICY[NodeID]])
            else:
                ax[0].plot(np.arange(avg_window*STEP, SIM_TIME, STEP), (data[avg_window:,NodeID]-data[:-avg_window,NodeID])/(avg_window*STEP), linewidth=5*REP[NodeID]/REP[0], color=colors[MODE[NodeID]])
                ax[1].plot(np.arange(avg_window*STEP, SIM_TIME, STEP), (data[avg_window:,NodeID]-data[:-avg_window,NodeID])*sum(REP)/(NU*REP[NodeID]*avg_window*STEP), linewidth=5*REP[NodeID]/REP[0], color=colors[MODE[NodeID]])
    
    ax[0].set_xlim(0, SIM_TIME)
    ax[1].set_xlim(0, SIM_TIME)
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)

    plt.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(dirstr+'/plots/Scaled '+ylabel+'.png', bbox_inches='tight')

def per_node_rate_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window: int = 1000, legend_loc: str = 'upper right', modes = None, step=STEP, figtxt = ''):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.title.set_text(title)

    if modes is None:
        modes = list(set(MODE))
    mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if MODE[NodeID] in modes and np.any(data[:, NodeID]):
            ax.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), (data[avg_window:,NodeID]-data[:-avg_window,NodeID])/(avg_window*STEP), color=colors[MODE[NodeID]])
    
    ax.set_xlim(0, SIM_TIME)
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)

    plt.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(dirstr+'/plots/'+ylabel+'.png', bbox_inches='tight')


def per_node_plotly_plot(time, data: np.ndarray, xlabel: str, ylabel: str, title: str, avg_window: int = 2000, legend_loc: str = 'upper right', modes = None, step=STEP):
    fig = go.Figure()
    max_val = np.amax(data)
    fig.update_layout(title=title,
                   xaxis_title=xlabel,
                   yaxis_title=ylabel,
                   yaxis_range=[0, 1.1*max_val])
    if modes is None:
        modes = list(set(MODE))
    colors = ['gray', 'blue', 'red', 'green']
    for NodeID in range(NUM_NODES):
        if np.any(data[:, NodeID]):
            fig.add_trace(go.Scatter(x=np.arange((avg_window-1)*step-SIM_TIME+time, time, step),
                                     y=np.convolve(np.ones(avg_window)/avg_window, data[:,NodeID], 'valid'),
                                     mode='lines',
                                     name="Node " + str(NodeID+1)))
    
    """ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)"""

    return fig

def all_node_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, interval=1):
    _, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.plot(np.arange(0, SIM_TIME, STEP*interval), data, color='black')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(dirstr, bbox_inches='tight')