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

def plot_cdf_exp(data, xlabel: str, dirstr: str,  xlim=0):
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
    h = []
    for NodeID in range(NUM_NODES):
        bins = np.arange(0, round(maxval*1/step), 1)*step
        pdf = np.zeros(len(bins))
        i = 0
        if not isinstance(data[NodeID][0], int):
            if data[NodeID][0].size>1:
                lats = sorted(data[NodeID][0])
        for lat in lats:
            while i<len(bins):
                if lat>=bins[i]:
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
        if MODE[NodeID]==0:
            Lines[NodeID] = ax.plot(bins, cdf, color='tab:gray', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
        if MODE[NodeID]==1:
            Lines[NodeID] = ax.plot(bins, cdf, color='tab:blue', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
        if MODE[NodeID]==2:
            Lines[NodeID] = ax.plot(bins, cdf, color='tab:red', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
        if MODE[NodeID]==3:
            mal = True
            Lines[NodeID] = ax.plot(bins, cdf, color='tab:green', linewidth=4*REP[NodeID]/REP[0], marker=marker, markevery=0.1)
        h.append(np.mean(data[NodeID][0]))
    h = sum(h)/len(h)
    h0 = h-0.5
    h1 = h+0.5
    mu = 1/h
    if mal:
        ModeLines = [Line2D([0],[0],color='tab:gray', lw=4), Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='tab:green', lw=4), Line2D([0],[0],color='black', lw=1, linestyle='dashdot'), Line2D([0],[0],color='black', lw=1, linestyle='dashed'), Line2D([0],[0],color='black', lw=1, linestyle='dotted')]
        ax.legend(ModeLines, ['Inactive', 'Content','Best-effort', 'Malicious', 'Exponential CDF', 'Uniform CDF', r'$h=\mu^{-1}=$'+"{:.2f}".format(h)], loc='lower right')
    elif iot:
        ModeLines = [Line2D([0],[0],color='tab:blue'), Line2D([0],[0],color='tab:red'), Line2D([0],[0],color='tab:blue', marker='x'), Line2D([0],[0],color='tab:red', marker='x')]
        ax.legend(ModeLines, ['Content value node','Best-effort value node', 'Content IoT node', 'Best-effort IoT node'], loc='lower right')
    else:
        ModeLines = [Line2D([0],[0],color='tab:gray', lw=4), Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='black', lw=1, linestyle='dashdot'), Line2D([0],[0],color='black', lw=1, linestyle='dashed'), Line2D([0],[0],color='black', lw=1, linestyle='dotted')]
        ax.legend(ModeLines, ['Inactive', 'Content','Best-effort', 'Exponential CDF', 'Uniform CDF', r'$h=\mu^{-1}=$'+"{:.2f}".format(h)], loc='lower right')
    ax.plot(bins, np.ones(len(bins))-np.exp(-mu*bins), color='black', linestyle='dashdot')
    ax.plot([0,h0,h1,bins[-1]], [0,0,1,1],  color='black', linestyle='dashed')
    ax.axvline(h, color='black', linestyle='dotted')
    plt.savefig(dirstr, bbox_inches='tight')

def plot_cdf_weib(data, k, xlabel: str, dirstr: str,  xlim=0):
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
        ModeLines = [Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='black', lw=1, linestyle='dashed')]
        ax.legend(ModeLines, ['Content','Best-effort', r'$1-e^{(\mu t)^k}$'], loc='lower right')
    lmd = np.mean(data[1][0])
    ax.plot(bins, np.ones(len(bins))-1/(np.exp(np.power((1/lmd)*bins,k))), color='black', linestyle='dashed')
    plt.savefig(dirstr, bbox_inches='tight')
    print(lmd)


def plot_ratesetter_comp(dir1, dir2, dir3):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel('Time (sec)')
    axt = ax.twinx()  
    ax.tick_params(axis='y', labelcolor='black')
    axt.tick_params(axis='y', labelcolor='tab:gray')
    ax.set_ylabel('Confirmation Rate (blocks/sec)', color='black')
    axt.set_ylabel('Confirmation Latency (sec)', color='tab:gray')
    
    avg_window=1000
    data = np.loadtxt(dir1+'/raw/Number of Confirmed Messages.csv', delimiter=',')
    avgTP1 = np.concatenate((np.zeros((avg_window, 20)),(data[avg_window:,:]-data[:-avg_window,:])))/(avg_window*STEP)
    avgMeanDelay1 = np.loadtxt(dir1+'/raw/avgConfDelay.csv', delimiter=',')
    avgOUA1 = np.loadtxt(dir1+'/raw/avgOldestUnconfAge.csv', delimiter=',')
    data = np.loadtxt(dir2+'/raw/Number of Confirmed Messages.csv', delimiter=',')
    avgTP2 = np.concatenate((np.zeros((avg_window, 40)),(data[avg_window:,:]-data[:-avg_window,:])))/(avg_window*STEP)
    avgMeanDelay2 = np.loadtxt(dir2+'/raw/avgConfDelay.csv', delimiter=',')
    avgOUA2 = np.loadtxt(dir2+'/raw/avgOldestUnconfAge.csv', delimiter=',')
    data = np.loadtxt(dir3+'/raw/Number of Confirmed Messages.csv', delimiter=',')
    avgTP3 = np.concatenate((np.zeros((avg_window, 60)),(data[avg_window:,:]-data[:-avg_window,:])))/(avg_window*STEP)
    avgMeanDelay3 = np.loadtxt(dir3+'/raw/avgConfDelay.csv', delimiter=',')
    avgOUA3 = np.loadtxt(dir3+'/raw/avgOldestUnconfAge.csv', delimiter=',')
    
    markerevery = 2000
    ax.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), np.sum(avgTP1[avg_window:,:], axis=1), color = 'black', marker = 'o', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 100*STEP), avgMeanDelay1, color='tab:gray', marker = 'o', markevery=int(markerevery/100)) 
    ax.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), np.sum(avgTP2[avg_window:,:], axis=1), color = 'black', marker = 'x', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 100*STEP), avgMeanDelay2, color='tab:gray', marker = 'x', markevery=int(markerevery/100))  
    ax.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), np.sum(avgTP3[avg_window:,:], axis=1), color = 'black', marker = '^', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 100*STEP), avgMeanDelay3, color='tab:gray', marker = '^', markevery=int(markerevery/100))  
    ax.set_ylim([0,350])
    axt.set_ylim([0,18])
    ax.axhline(NU, xmax=SIM_TIME, color='tab:red', linestyle='--')
    
    ModeLines1 = [Line2D([0],[0],color='black', linestyle='None', marker='o'), Line2D([0],[0],color='black', linestyle='None', marker='x'), Line2D([0],[0],color='black', linestyle='None', marker='^')]
    ModeLines2 = [Line2D([0],[0],color='black'), Line2D([0],[0],color='tab:red', linestyle='--'), Line2D([0],[0],color='tab:gray')]
    
    ax.legend(ModeLines1, [r'$|\mathcal{M}|=20$', r'$|\mathcal{M}|=40$', r'$|\mathcal{M}|=60$'], loc='lower center', ncol=1)
    axt.legend(ModeLines2, ['Confirmation Rate', 'Scheduling Rate', 'Confirmation Latency'], loc='lower right', ncol=1)
    
    dirstr = os.path.dirname(os.path.realpath(__file__)) + '/results/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    
    copyfile(dir1+'/global_params.txt', dirstr+'/config1.txt')
    copyfile(dir2+'/global_params.txt', dirstr+'/config2.txt')
    copyfile(dir3+'/global_params.txt', dirstr+'/config3.txt')
    
    fig.tight_layout()
    plt.savefig(dirstr+'/Throughput.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Max time partially confirmed (sec)')
    ax.plot(np.arange((avg_window-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(avg_window)/avg_window, avgOUA1, 'valid'), color = 'black', marker = 'o', markevery=markerevery)
    ax.plot(np.arange((avg_window-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(avg_window)/avg_window, avgOUA2, 'valid'), color = 'black', marker = 'x', markevery=markerevery)
    ax.plot(np.arange((avg_window-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(avg_window)/avg_window, avgOUA3, 'valid'), color = 'black', marker = '^', markevery=markerevery)
    ax.legend(ModeLines1, [r'$|\mathcal{M}|=20$', r'$|\mathcal{M}|=40$', r'$|\mathcal{M}|=60$'], loc='lower right', ncol=1)

    fig.tight_layout()
    plt.savefig(dirstr+'/OUA.png', bbox_inches='tight')

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
    #ax.set_xticks([0,5,10,15])
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

def per_node_plot_mean(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window: int = 100, legend_loc: str = 'right', modes = None, step=STEP, figtxt = ''):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.title.set_text(title)

    avg = np.mean(data[int(20/STEP):,0])

    if modes is None:
        modes = list(set(MODE))
    mode_names = ['Inactive', 'Content','Best-effort', 'Malicious', 'Multi-rate']
    colors = ['tab:gray', 'tab:blue', 'tab:red', 'tab:green', 'tab:olive']
    for NodeID in range(NUM_NODES):
        if MODE[NodeID] in modes and np.any(data[:, NodeID]):
            ax.plot(np.arange((avg_window-1)*step, SIM_TIME, step), np.convolve(np.ones(avg_window)/avg_window, data[:,NodeID], 'valid'), color=colors[MODE[NodeID]])
    ax.axhline(avg, color='black', linestyle='dotted')
    ax.set_xlim(0, SIM_TIME)
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes]
    ModeLines.append(Line2D([0],[0],color='black', linestyle='dotted'))
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[i] for i in modes]+[r'$L=$'+'{:03d}'.format(int(avg))], loc=legend_loc)

    plt.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(dirstr+'/plots/'+ylabel+'.png', bbox_inches='tight')

def scaled_rate_plot(data: np.ndarray, xlabel: str, ylabel1: str, ylabel2: str, title: str, dirstr: str, avg_window: int = 1000, legend_loc: str = 'right', modes = None, step=STEP, figtxt = ''):
    
    fig, ax = plt.subplots(2,1, sharex=True, figsize=(8,8))
    ax[0].grid(linestyle='--')
    ax[1].grid(linestyle='--')
    ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel1)
    ax[1].set_ylabel(ylabel2)

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
    ModeLines = [Line2D([0],[0],color=colors[mode], lw=4) for mode in modes if mode!=0]
    if len(modes)>1:
        fig.legend(ModeLines, [mode_names[mode] for mode in modes if mode!=0], loc=legend_loc)

    plt.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.savefig(dirstr+'/plots/Scaled '+title+'.png', bbox_inches='tight')

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

def all_node_plot(data: np.ndarray, xlabel: str, ylabel: str, title: str, dirstr: str, avg_window=1):
    _, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.plot(np.arange((avg_window-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(avg_window)/avg_window, data, 'valid'), color='black')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(dirstr, bbox_inches='tight')