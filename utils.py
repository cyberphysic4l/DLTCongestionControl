from core.global_params import *
from matplotlib.lines import Line2D
from shutil import copyfile
import os
import matplotlib.pyplot as plt
from time import gmtime, strftime
import plotly.graph_objects as go


def plot_cdf(data, xlabel: str, dirstr: str,  xlim=0):
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
    Lines = [[] for NodeID in range(NUM_NODES)]
    mal = False
    iot = False
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
                iot = True
                marker = 'x'
            else:
                marker = None
            if MODE[NodeID]==1:
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:blue', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
            if MODE[NodeID]==2:
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:red', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
            if MODE[NodeID]==3:
                mal = True
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:green', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
    if mal:
        ModeLines = [Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:green', lw=4)]
        ax.legend(ModeLines, ['Best-effort', 'Content','Malicious'], loc='lower right')
    elif iot:
        ModeLines = [Line2D([0],[0],color='tab:blue'), Line2D([0],[0],color='tab:red'), Line2D([0],[0],color='tab:blue', marker='x'), Line2D([0],[0],color='tab:red', marker='x')]
        ax.legend(ModeLines, ['Content value node','Best-effort value node', 'Content IoT node', 'Best-effort IoT node'], loc='lower right')
    else:
        ModeLines = [Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4)]
        ax.legend(ModeLines, ['Content','Best-effort'], loc='lower right')
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


def plot_ratesetter_comp(dir1, dir2, dir3):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel('Time (sec)')
    axt = ax.twinx()  
    ax.tick_params(axis='y', labelcolor='black')
    axt.tick_params(axis='y', labelcolor='tab:gray')
    ax.set_ylabel(r'$DR/\nu \quad (\%)$', color='black')
    axt.set_ylabel('Mean Latency (sec)', color='tab:gray')
    
    avgTP1 = np.loadtxt(dir1+'/avgTP.csv', delimiter=',')
    avgMeanDelay1 = np.loadtxt(dir1+'/avgMeanDelay.csv', delimiter=',')
    avgTP2 = np.loadtxt(dir2+'/avgTP.csv', delimiter=',')
    avgMeanDelay2 = np.loadtxt(dir2+'/avgMeanDelay.csv', delimiter=',')
    avgTP3 = np.loadtxt(dir3+'/avgTP.csv', delimiter=',')
    avgMeanDelay3 = np.loadtxt(dir3+'/avgMeanDelay.csv', delimiter=',')
    
    markerevery = 500
    ax.plot(np.arange(10, SIM_TIME, STEP), 100*np.sum(avgTP1[1000:,:]/NU, axis=1), color = 'black', marker = 'o', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay1, color='tab:gray', marker = 'o', markevery=int(markerevery*STEP)) 
    ax.plot(np.arange(10, SIM_TIME, STEP), 100*np.sum(avgTP2[1000:,:]/NU, axis=1), color = 'black', marker = 'x', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay2, color='tab:gray', marker = 'x', markevery=int(markerevery*STEP))  
    ax.plot(np.arange(10, SIM_TIME, STEP), 100*np.sum(avgTP3[1000:,:]/NU, axis=1), color = 'black', marker = '^', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay3, color='tab:gray', marker = '^', markevery=int(markerevery*STEP))  
    ax.set_ylim([0,110])
    axt.set_ylim([0,9])
    
    ModeLines = [Line2D([0],[0],color='black', linestyle=None, marker='o'), Line2D([0],[0],color='black', linestyle=None, marker='x'), Line2D([0],[0],color='black', linestyle=None, marker='^')]
    #ax.legend(ModeLines, [r'$A=0.05$', r'$A=0.075$', r'$A=0.1$'], loc='lower right')
    #ax.set_title(r'$\beta=0.7, \quad W=2$')
    #ax.legend(ModeLines, [r'$\beta=0.5$', r'$\beta=0.7$', r'$\beta=0.9$'], loc='lower right')
    #ax.set_title(r'$A=0.075, \quad W=2$')
    ax.legend(ModeLines, [r'$W=1$', r'$W=2$', r'$W=3$'], loc='lower right')
    ax.set_title(r'$A=0.075, \quad \beta=0.7$')
    #ax.legend(ModeLines, ['Our algorithm', 'PoW case 1', 'PoW case 2'], loc='right')
    #ax.set_title('Our algorithm vs. PoW')
    #ax.legend(ModeLines, [r'$|\mathcal{M}|=25$', r'$|\mathcal{M}|=50$', r'$|\mathcal{M}|=75$'], loc='lower right', ncol=1)
    #ax.set_title(r'$A=0.075, \quad \beta=0.7, \quad W=2$')
    #ax.legend(ModeLines, ['PoW case 1', 'PoW case 2', 'PoW case 3'], loc='right')
    
    dirstr = os.path.dirname(os.path.realpath(__file__)) + '/results/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    
    copyfile(dir1+'/aaconfig.txt', dirstr+'/config1.txt')
    copyfile(dir2+'/aaconfig.txt', dirstr+'/config2.txt')
    copyfile(dir3+'/aaconfig.txt', dirstr+'/config3.txt')
    
    fig.tight_layout()
    plt.savefig(dirstr+'/Throughput.png', bbox_inches='tight')
    

def plot_scheduler_comp(dir1, dir2):
    
    latencies1 = []
    for NodeID in range(NUM_NODES):
        if os.stat(dir1+'/latencies'+str(NodeID)+'.csv').st_size != 0:
            lat = [np.loadtxt(dir1+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            lat = [0]
        latencies1.append(lat)
        
    latencies2 = []
    for NodeID in range(NUM_NODES):
        if os.stat(dir2+'/latencies'+str(NodeID)+'.csv').st_size != 0:
            lat = [np.loadtxt(dir2+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            lat = [0]
        latencies2.append(lat)
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(8,4))
    ax[0].grid(linestyle='--')
    ax[1].grid(linestyle='--')
    ax[1].set_xlabel('Latency (sec)')
    ax[0].set_title('DRR')
    ax[1].set_title('DRR-')
    xlim = plot_cdf(latencies1, ax[0])
    plot_cdf(latencies2, ax[1], xlim)
    
    dirstr = os.path.dirname(os.path.realpath(__file__)) + '/results/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    
    copyfile(dir1+'/aaconfig.txt', dirstr+'/config1.txt')
    copyfile(dir2+'/aaconfig.txt', dirstr+'/config2.txt')
    plt.savefig(dirstr+'/LatencyComp.png', bbox_inches='tight')

def per_node_barplot(data, xlabel: str, ylabel: str, title: str, dirstr: str, legend_loc: str = 'upper right', modes=None):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.title.set_text(title)
    ax.set_ylabel(ylabel)
    if modes is None:
        modes = list(set(MODE))
    mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        ax.bar(NodeID, data[NodeID], color=colors[MODE[NodeID]])
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes], loc=legend_loc)
    plt.savefig(dirstr, bbox_inches='tight')

def per_node_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window: int = 100, legend_loc: str = 'upper right', modes = None, step=STEP, figtxt = ''):
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
            ax.plot(np.arange((avg_window-1)*step, SIM_TIME, step), np.convolve(np.ones(avg_window)/avg_window, data[:,NodeID], 'valid'), color=colors[MODE[NodeID]])
    
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

    if modes is None:
        modes = list(set(MODE))
    mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if MODE[NodeID] in modes and np.any(data[:, NodeID]):
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

def all_node_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str):
    _, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.plot(np.arange(0, SIM_TIME, STEP), data, color='black')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(dirstr, bbox_inches='tight')