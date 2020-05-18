# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
from random import sample
from pathlib import Path

np.random.seed(0)
# Simulation Parameters
MONTE_CARLOS = 100
SIM_TIME = 180
STEP = 0.01
# Network Parameters
NU = 10
NUM_NODES = 15
NUM_NEIGHBOURS = 4
START_TIMES = 10*np.ones(NUM_NODES)
TB = True
REP = np.ones(NUM_NODES, dtype=int)
REP[0] = 3
REP[1] = 2
REP[5] = 3
REP[6] = 2
REP[10] = 3
REP[11] = 2

MODE = np.zeros(NUM_NODES, dtype=int) # inactive
MODE[5:10] = 1 # content
MODE[10:15] = 2 # best-effort
MODE[14] = 3 # malicious

# Congestion Control Parameters
ALPHA = 0.1
BETA = 0.5
WAIT_TIME = 2
MAX_INBOX_LEN = 2
MAX_BURST = 1
QUANTUM = 1
MAX_TOKENS = 1

SCHEDULING = 'drr'
    
def main():
    '''
    Create directory for storing results with these parameters
    '''
    dirstr = 'data/globecom/nu='+str(NU)+'/nmc='+str(MONTE_CARLOS)+'/rep='+''.join(str(int(e)) for e in REP)+'/mode='+''.join(str(int(e)) for e in MODE)+'/10sec'
    #dirstr = 'data/sched='+SCHEDULING+'/tb='+str(TB)+'/dmax='+str(MAX_BURST)+'/nu='+str(NU)+'/rep='+''.join(str(int(e)) for e in REP)+'/mode='+''.join(str(int(e)) for e in MODE)+'/alpha='+str(ALPHA)+'/beta='+str(BETA)+'/tau='+str(WAIT_TIME)+'/inbox='+str(MAX_INBOX_LEN)+'/neighbours='+str(NUM_NEIGHBOURS)+'/simtime='+str(SIM_TIME)+'/nmc='+str(MONTE_CARLOS)+'/undissem'
    if not Path(dirstr).exists():
        print("Simulating")
        simulate(dirstr)
    else:
        print("Simulation already done for these parameters")
        #simulate(dirstr)
    plot_results(dirstr)
    
def simulate(dirstr):
    """
    Setup simulation inputs and instantiate output arrays
    """
    # seed rng
    np.random.seed(0)
    TimeSteps = int(SIM_TIME/STEP)
    
    """
    Monte Carlo Sims
    """
    Lmds = []
    InboxLens = []
    Deficits = []
    SymDiffs = []
    Throughput = []
    Undissem = []
    MeanDelay = []
    Util = []
    latencies = [[] for NodeID in range(NUM_NODES)]
    latTimes = [[] for NodeID in range(NUM_NODES)]
    for mc in range(MONTE_CARLOS):
        """
        Generate network topology:
        Comment out one of the below lines for either random k-regular graph or a
        graph from an adjlist txt file i.e. from the autopeering simulator
        """
        G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES)
        #G = nx.read_adjlist('input_adjlist.txt', delimiter=' ')
        # Get adjacency matrix and weight by delay at each channel
        ChannelDelays = 0.05*np.ones((NUM_NODES, NUM_NODES))+0.1*np.random.rand(NUM_NODES, NUM_NODES)
        AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
        Net = Network(AdjMatrix)
        # output arrays
        Lmds.append(np.zeros((TimeSteps, NUM_NODES)))
        InboxLens.append(np.zeros((TimeSteps, NUM_NODES)))
        Deficits.append(np.zeros((TimeSteps, NUM_NODES)))
        SymDiffs.append(np.zeros(TimeSteps))
        Throughput.append(np.zeros((TimeSteps, NUM_NODES)))
        Undissem.append(np.zeros((TimeSteps, NUM_NODES)))
        for i in range(TimeSteps):
            if 100*i/TimeSteps%10==0:
                print("Simulation: "+str(mc) +"\t " + str(int(100*i/TimeSteps))+"% Complete")
            # discrete time step size specified by global variable STEP
            T = STEP*i
            """
            The next line is the function which ultimately calls all others
            and runs the simulation for a time step
            """
            Net.simulate(T)
            # save summary results in output arrays
            for NodeID in range(NUM_NODES):
                Lmds[mc][i, NodeID] = Net.Nodes[NodeID].Lambda
                InboxLens[mc][i, NodeID] = len(Net.Nodes[0].Inbox.Packets[NodeID])
                Deficits[mc][i, NodeID] = Net.Nodes[0].Inbox.Deficit[NodeID]
                Throughput[mc][i, NodeID] = Net.Throughput[NodeID]
                Undissem[mc][i,NodeID] = Net.Nodes[NodeID].Undissem
            #SymDiffs[mc][i] = Net.sym_diffs().max()
        
        MeanDelay.append(np.zeros(SIM_TIME))
        for NodeID in range(NUM_NODES):
            for i in range(SIM_TIME):
                delays = [Net.TranDelays[j] for j in range(len(Net.TranDelays)) if int(Net.DissemTimes[j])==i]
                if delays:
                    MeanDelay[mc][i] = sum(delays)/len(delays)
                
        latencies, latTimes = Net.tran_latency(latencies, latTimes)
        """
        for NodeID in range(NUM_NODES):
            if latencies[NodeID]:
                for i in range(SIM_TIME):
                    if latTimes[NodeID][]
           """     
        Util.append(np.concatenate((np.zeros((1000, NUM_NODES)),(Throughput[mc][1000:,:]-Throughput[mc][:-1000,:])))/10)
        #Util.append(np.convolve(np.zeros((Throughput[mc][500:,:]-Throughput[mc][:-500,:])))/5)
        del Net
    """
    Get results
    """
    avgLmds = sum(Lmds)/len(Lmds)
    avgUtil = sum(Util)/len(Util)
    avgInboxLen = sum(InboxLens)/len(InboxLens)
    avgDefs = sum(Deficits)/len(Deficits)
    avgMSDs = sum(SymDiffs)/len(SymDiffs)
    avgUndissem = sum(Undissem)/len(Undissem)
    avgMeanDelay = sum(MeanDelay)/len(MeanDelay)
    """
    Create a directory for these results and save them
    """
    Path(dirstr).mkdir(parents=True, exist_ok=True)
    np.savetxt(dirstr+'/avgMSDs.csv', avgMSDs, delimiter=',')
    np.savetxt(dirstr+'/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/avgUtil.csv', avgUtil, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLen.csv', avgInboxLen, delimiter=',')
    np.savetxt(dirstr+'/avgDefs.csv', avgDefs, delimiter=',')
    np.savetxt(dirstr+'/avgUndissem.csv', avgUndissem, delimiter=',')
    np.savetxt(dirstr+'/avgMeanDelay.csv', avgMeanDelay, delimiter=',')
    for NodeID in range(NUM_NODES):
        np.savetxt(dirstr+'/latTimes'+str(NodeID)+'.csv', np.asarray(latTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/latencies'+str(NodeID)+'.csv', np.asarray(latencies[NodeID]), delimiter=',')
    nx.write_adjlist(G, dirstr+'/result_adjlist.txt', delimiter=' ')
    
def plot_results(dirstr):
    """
    Initialise plots
    """
    plt.close('all')
    
    """
    Load results from the data directory
    """
    avgMSDs = np.loadtxt(dirstr+'/avgMSDs.csv', delimiter=',')
    avgLmds = np.loadtxt(dirstr+'/avgLmds.csv', delimiter=',')
    avgUtil = np.loadtxt(dirstr+'/avgUtil.csv', delimiter=',')
    avgInboxLen = np.loadtxt(dirstr+'/avgInboxLen.csv', delimiter=',')
    avgUndissem = np.loadtxt(dirstr+'/avgUndissem.csv', delimiter=',')
    avgMeanDelay = np.loadtxt(dirstr+'/avgMeanDelay.csv', delimiter=',')
    latencies = []
    latTimes = []
    
    for NodeID in range(NUM_NODES):
        lat = [np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        if lat:
            latencies.append(lat)
        tLat = [np.loadtxt(dirstr+'/latTimes'+str(NodeID)+'.csv', delimiter=',')]
        if tLat:
            latTimes.append(tLat)
    """
    Plot results
    """
    fig1, ax1 = plt.subplots(3,1, sharex=True, figsize=(8,8))
    fig2, ax2 = plt.subplots(figsize=(8,4))
    fig3, ax3 = plt.subplots(figsize=(8,4))
    fig4, ax4 = plt.subplots(figsize=(8,4))
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax1[0].title.set_text('Dissemination Rate')
    ax1[1].title.set_text('Scaled Dissemination Rate')
    ax1[2].title.set_text('# Undisseminated Transactions')
    
    ax1[0].grid(linestyle='--')
    ax1[1].grid(linestyle='--')
    ax1[2].grid(linestyle='--')    
    ax2.grid(linestyle='--')   
    ax3.grid(linestyle='--')
    ax1[2].set_xlabel('Time (sec)')
    ax4.set_xlabel('Time (sec)')
    ax5.set_xlabel('Time (sec)')
    ax2.set_xlabel('Time (sec)')
    ax3.set_xlabel('Latency (sec)')
    #ax1[0].set_ylabel(r'${\lambda_i} / {\~{\lambda}_i}$')
    ax1[0].set_ylabel(r'$D_i$')
    ax1[1].set_ylabel(r'$D_i / {\~{\lambda}_i}$')    
    ax1[2].set_ylabel(r'$U_i$')
    ax4.set_ylabel(r'$\lambda_i$')
    ax5.set_ylabel('Mean Delay')
    opp = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=15, label='Opportunistic')
    cont = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=15, label='Content')
    Lines = [[] for NodeID in range(NUM_NODES)]
    ax2.plot(np.arange(10, SIM_TIME, STEP), np.sum(avgUtil[1000:,:], axis=1), color = 'tab:blue')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay, color='tab:red')    
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    ax22.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylabel(r'$D$', color='tab:blue')
    ax22.set_ylabel('Mean Delay', color='tab:red')
    fig2.tight_layout()
    #ax2.plot([0, SIM_TIME], [NU, NU], 'r--', linewidth=0.5)
    lw = 1
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            if REP[NodeID]==3:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:blue')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:blue')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:blue')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:blue')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:blue')
            if REP[NodeID]==2:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:orange')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:orange')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:orange')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:orange')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:orange')
            if REP[NodeID]==1:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:green')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:green')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:green')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:green')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:green')
        if MODE[NodeID]==2:
            if REP[NodeID]==3:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:red')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:red')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:red')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:red')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:red')
            if REP[NodeID]==2:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:purple')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:purple')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:purple')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:purple')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:purple')
            if REP[NodeID]==1:
                #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:brown')
                ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:brown')
                ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:brown')
                ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:brown')
                Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:brown')
        if MODE[NodeID]==3:
            #Lines[NodeID] = ax1[0].plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID]*sum(REP)/(NU*REP[NodeID]), color='tab:gray')
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=lw, color='tab:gray')
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID], linewidth=lw, color='tab:gray')
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgUtil[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=lw, color='tab:gray')
            Lines[NodeID] = ax1[2].plot(np.arange(10, SIM_TIME+STEP, STEP), np.convolve(avgUndissem[:,NodeID], np.ones((1000,))/1000, mode='valid'), color='tab:gray')
    
    # Latency
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(latencies[NodeID][0])>0:
            val = np.max(latencies[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    Lines = [[] for NodeID in range(NUM_NODES)]
    for NodeID in range(len(latencies)):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval), STEP)
            pdf = np.zeros(len(bins))
            for i, b in enumerate(bins):
                lats = [lat for lat in latencies[NodeID][0] if (lat>b and lat<b+STEP)]
                pdf[i] = len(lats)
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            if MODE[NodeID]==1 and REP[NodeID]==3:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:blue')
            if MODE[NodeID]==1 and REP[NodeID]==2:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:orange')
            if MODE[NodeID]==1 and REP[NodeID]==1:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:green')
            if MODE[NodeID]==2 and REP[NodeID]==3:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:red')
            if MODE[NodeID]==2 and REP[NodeID]==2:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:purple')
            if MODE[NodeID]==2 and REP[NodeID]==1:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:brown')
            if MODE[NodeID]==3:
                Lines[NodeID] = ax3.plot(bins, cdf, color='tab:gray')
    # legends
    if MODE[14]==3:
        fig1.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0], Lines[14][0]), ('C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'M1'), loc=5, ncol=1)
        ax3.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0], Lines[14][0]), ('C3', 'C2', 'C1', 'B3', 'B2', 'B1', 'M1'), loc=5, ncol=1)

    else:
        #fig1.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0]), (r'$i = 6$', r'$i = 7$', r'$i = 8-10$', r'$i = 11$', r'$i = 12$', r'$i = 13-15$'), loc=5, ncol=1)
        #ax3.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0]), (r'$i = 6$', r'$i = 7$', r'$i = 8-10$', r'$i = 11$', r'$i = 12$', r'$i = 13-15$'), loc=5, ncol=1)
        fig1.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0]), ('C3', 'C2', 'C1', 'B3', 'B2', 'B1'), loc=5, ncol=1)
        ax3.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0]), ('C3', 'C2', 'C1', 'B3', 'B2', 'B1'), loc=5, ncol=1)
    #ax1[0].set_ylim((0,6))
    #ax1[1].set_ylim((0,6))
    #ax2.set_ylim((0,12))
    
    plt.figure(fig1.number)
    plt.savefig(dirstr+'/Rates.png', bbox_inches='tight')
    """
    ax1[0].plot([0, SIM_TIME], [2, 2], 'r--', linewidth=0.5)
    ax1[0].plot([0, SIM_TIME], [1, 1], 'r--', linewidth=0.5)
    #ax1[1].plot([0, SIM_TIME], [2, 2], 'r--', linewidth=0.5)
    #ax1[1].plot([0, SIM_TIME], [1, 1], 'r--', linewidth=0.5)
    plt.figure(fig1.number)
    plt.savefig(dirstr+'/Rates_rdashed.png')
    """ 
    plt.figure(fig2.number)
    plt.savefig(dirstr+'/Throughput.png', bbox_inches='tight')
    
    plt.figure(fig3.number)
    plt.savefig(dirstr+'/Latency.png', bbox_inches='tight')
    
    
    
    """
    ax1.set_xlabel('Time (sec)')
    ax1.set_ylabel('Throughput')
    opp = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                          markersize=15, label='Opportunistic')
    cont = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
                          markersize=15, label='Content')
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==2:
            if REP[NodeID]==3:
                ax1.plot(np.arange(0, SIM_TIME, STEP), np.convolve(avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), np.ones((10,))/10, mode='same'), linewidth=0.5)#, linestyle='-', marker='x', markevery=500)#, color='red')
                #ax1.plot(np.arange(0, SIM_TIME, STEP), avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=0.5, linestyle=':', color='red')
            if REP[NodeID]==2:
                ax1.plot(np.arange(0, SIM_TIME, STEP), np.convolve(avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), np.ones((10,))/10, mode='same'), linewidth=0.5)#, linestyle='-', marker='x', markevery=500)#, color='green')
                #ax1.plot(np.arange(0, SIM_TIME, STEP), avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=0.5, linestyle=':', color='green')
            if REP[NodeID]==1:
                ax1.plot(np.arange(0, SIM_TIME, STEP), np.convolve(avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), np.ones((10,))/10, mode='same'), linewidth=0.5)#, linestyle='-', marker='x', markevery=500)#, color='blue')
                #ax1.plot(np.arange(0, SIM_TIME, STEP), avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=0.5, linestyle=':', color='blue')
        if MODE[NodeID]==3:
            ax1.plot(np.arange(0, SIM_TIME, STEP), np.convolve(avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), np.ones((10,))/10, mode='same'), linewidth=0.5)#, linestyle='-', marker='x', markevery=500)
            #ax1.plot(np.arange(0, SIM_TIME, STEP), avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=0.5, linestyle=':', color='black')
        if NodeID==5:
            ax1.plot(np.arange(0, SIM_TIME, STEP), np.convolve(avgUtil[:,NodeID]*sum(REP)/(NU*REP[NodeID]), np.ones((10,))/10, mode='same'), linewidth=0.5)#, linestyle='-', marker='x', markevery=500, color='black')
            
    ax1.legend(('Nodes 6--10', 'Node 11','Node 12','Node 13','Node 14','Node 15'), loc=1)
    ax1.plot([0, SIM_TIME], [2, 2], 'r--')
    #ax1.plot([0, SIM_TIME], [1, 1], 'r--'))
    plt.savefig(dirstr+'/Throughput.png')
    
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Time (sec)')
    ax2.set_ylabel('Max Symmetric Difference')
    ax2.plot(np.arange(0, SIM_TIME, STEP), avgMSDs)
    plt.savefig(dirstr+'/SymDif.png')
    
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Network Throughput (tx/sec)')
    y = np.sum(avgUtil, axis=1)
    ax3.plot(np.arange(0, SIM_TIME, STEP), np.convolve(y, np.ones((10,))/10, mode='same'))
    ax3.plot([0, SIM_TIME], [NU, NU], 'r--')
    plt.savefig(dirstr+'/Util.png')

    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Undisseminated Transactions')
    ax4.plot(np.arange(0, SIM_TIME, STEP), avgUndissem)
    plt.legend(range(NUM_NODES), loc=1)
    plt.savefig(dirstr+'/Undissem.png')
    
    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel('Inbox length at Node 0')
    ax4.plot(np.arange(0, SIM_TIME, STEP), avgInboxLen)
    plt.legend(range(NUM_NODES), loc=1)
    plt.savefig(dirstr+'/Inbox.png')
    
    fig5, ax5 = plt.subplots()
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(latencies[NodeID][0])>0:
            val = np.max(latencies[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    Lines = [[] for NodeID in range(NUM_NODES)]
    for NodeID in range(len(latencies)):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval), STEP)
            pdf = np.zeros(len(bins))
            for i, b in enumerate(bins):
                lats = [lat for lat in latencies[NodeID][0] if (lat>b and lat<b+STEP)]
                pdf[i] = len(lats)
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            if MODE[NodeID]==1 and REP[NodeID]==3:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:blue')
            if MODE[NodeID]==1 and REP[NodeID]==2:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:orange')
            if MODE[NodeID]==1 and REP[NodeID]==1:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:green')
            if MODE[NodeID]==2 and REP[NodeID]==3:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:red')
            if MODE[NodeID]==2 and REP[NodeID]==2:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:purple')
            if MODE[NodeID]==2 and REP[NodeID]==1:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:brown')
            if MODE[NodeID]==3:
                Lines[NodeID] = ax5.plot(bins, cdf, linewidth=0.5, color='tab:gray')
             #   ax5.plot(bins, cdf, color='red')
    ax5.set_xlabel('Latency (sec)')
    ax5.legend((Lines[5][0], Lines[6][0], Lines[7][0], Lines[10][0], Lines[11][0], Lines[12][0]), (r'$i = 6$', r'$i = 7$', r'$i = 8-10$', r'$i = 11$', r'$i = 12$', r'$i = 13-15$'), loc=5)
    plt.savefig(dirstr+'/Latency.png')
    
    fig6, ax6 = plt.subplots()
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(IATimes[NodeID][0])>0:
            val = np.max(IATimes[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    
    for NodeID in range(len(IATimes)):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval), STEP)
            pdf = np.zeros(len(bins))
            for i, b in enumerate(bins):
                lats = [lat for lat in IATimes[NodeID][0] if (lat>b and lat<b+STEP)]
                pdf[i] = len(lats)
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            ax6.plot(bins, cdf)
    plt.legend(range(NUM_NODES), loc=1)
    plt.savefig(dirstr+'/IATimes.png')
    """
    """
    Draw network graph used in this simulation
    """
    """
    G = nx.read_adjlist(dirstr+'/result_adjlist.txt', delimiter=' ')
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos)#, node_color=colors[0:NUM_NODES])
    plt.show()
    """

class Transaction:
    """
    Object to simulate a transaction its edges in the DAG
    """
    def __init__(self, ArrivalTime, Parents, Node, Index=None):
        self.ArrivalTime = ArrivalTime
        self.IssueTime = []
        self.Children = []
        self.Parents = Parents
        self.Index = Index
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
            self.InformedNodes = 0 # info about who has seen this TX
            self.GlobalSolidTime = []
        else: # genesis
            self.NodeID = []
            self.InformedNodes = 0 
            self.GlobalSolidTime = []
        
    def is_tip(self):
        if not self.Children:
            return True
        else:
            return False

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Ledger = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox()
        self.NodeID = NodeID
        self.Alpha = ALPHA*REP[NodeID]/sum(REP)
        self.Lambda = NU*REP[NodeID]/sum(REP)
        self.BackOff = []
        self.LastBackOff = []
        self.ArrivalTimes = [[] for NodeID in range(NUM_NODES)]
        self.LastWriteTime = 0
        self.LastTokenTime = 0
        self.Tokens = 0
        self.Bucket = []
        self.Undissem = 0
        
    def generate_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        if MODE[self.NodeID]>0:
            if TB and MODE[self.NodeID]==2:
                while Time >= self.LastTokenTime + 1/self.Lambda:
                    self.Tokens = min(self.Tokens+1, MAX_TOKENS)
                    self.LastTokenTime += 1/self.Lambda
                nTX = 5*REP[self.NodeID]-len(self.Bucket)
                if nTX>0:
                    NewTXs = np.sort(np.random.uniform(Time, Time+STEP, nTX))
                else:
                    NewTXs = []
            else:
                NewTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda)))
        else:
            NewTXs = []
            self.Lambda = 0
        for t in NewTXs:
            Parents = self.select_tips()
            self.TempTransactions.append(Transaction(t, Parents, self))
        # check PoW completion
        if self.TempTransactions:
            for Tran in self.TempTransactions:
                t = Tran.ArrivalTime + self.PoWDelay
                if t <= Time: # if PoW done
                    self.TempTransactions.remove(Tran)
                    # add the TX to Inbox as though it came from a virtual neighbour
                    p = Packet(self, self, Tran, t)
                    p.EndTime = Tran.ArrivalTime + self.PoWDelay
                    if MODE[self.NodeID]==3: # malicious don't consider own txs for scheduling
                        Tran.IssueTime = Tran.ArrivalTime + self.PoWDelay
                        self.add_to_ledger(self, Tran, t)
                    elif TB and MODE[self.NodeID]==2:
                        self.Bucket.append(p)
                    else:
                        self.add_to_inbox(p, Tran.ArrivalTime+self.PoWDelay)
        while self.Bucket and self.Tokens:
            self.add_to_inbox(self.Bucket.pop(0), Time)
            self.Tokens -= 1
    
    def select_tips(self):
        """
        Implements uniform random tip selection
        """
        if len(self.TipsSet)>1:
            Selection = sample(self.TipsSet, 2)
        else:
            Selection = self.Ledger[-2:-1]
        return Selection
    
    def schedule_txs(self, Time):
        """
        schedule txs from inbox at a fixed deterministic rate NU
        """
        # sort inboxes by arrival time
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        #nTX = np.random.poisson(STEP*NU)
        #times = np.sort(np.random.uniform(Time, Time+STEP, nTX))
        #i = 0
        while self.LastWriteTime+(1/NU)<Time+STEP:
            if SCHEDULING=='drr':
                Packet = self.Inbox.drr_schedule()
            elif SCHEDULING=='fifo':
                Packet = self.Inbox.fifo_schedule(Time)
            elif SCHEDULING=='aimd':
                Packet = self.Inbox.aimd_schedule(Time)
            elif SCHEDULING=='tokenbucket':
                Packet = self.Inbox.token_bucket_schedule(Time)
            elif SCHEDULING=='wrr':
                Packet = self.Inbox.wrr_schedule(Time)
            elif SCHEDULING=='brr':
                Packet = self.Inbox.brr_schedule(Time)
            elif SCHEDULING=='bob':
                Packet = self.Inbox.bob_schedule(Time)
            if Packet is not None:
                if Packet.Data not in self.Ledger:
                    self.add_to_ledger(Packet.TxNode, Packet.Data, max(Time, self.LastWriteTime+(1/NU)))
                # update AIMD
                self.set_rate(Time)
                self.LastWriteTime = max(Time, self.LastWriteTime+(1/NU))
            else:
                break
    
    def add_to_ledger(self, TxNode, Tran, Time):
        """
        Adds the transaction to the local copy of the ledger and broadcast it
        """
        self.Ledger.append(Tran)
        if Tran.NodeID==self.NodeID:
            self.Undissem += 1
        # mark this TX as received by this node
        Tran.InformedNodes += 1
        if Tran.InformedNodes==NUM_NODES:
            self.Network.Throughput[Tran.NodeID] += 1
            self.Network.TranDelays.append(Time-Tran.IssueTime)
            self.Network.DissemTimes.append(Time)
            Tran.GlobalSolidTime = Time
            self.Network.Nodes[Tran.NodeID].Undissem -= 1
        if not Tran.Children:
            self.TipsSet.append(Tran)
        if Tran.Parents:
            for Parent in Tran.Parents:
                Parent.Children.append(Tran)
                if Parent in self.TipsSet:
                    self.TipsSet.remove(Parent)
                else:
                    continue
        # broadcast the packet
        self.Network.broadcast_data(self, TxNode, Tran, Time)
    
    def check_congestion(self, Time):
        """
        Check if congestion is occurring
        """
        if self.Inbox.AllPackets:
            if len(self.Inbox.Packets[self.NodeID])>MAX_INBOX_LEN*REP[self.NodeID]:
                self.BackOff = True
            
    def set_rate(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2 and Time>=START_TIMES[self.NodeID]: # AIMD starts after 1 min adjustment
                # if wait time has not passed---reset.
                if self.LastBackOff:
                    if Time < self.LastBackOff + WAIT_TIME:
                        self.BackOff = False
                        return
                # multiplicative decrease or else additive increase
                if self.BackOff:
                    self.Lambda = self.Lambda*BETA
                    self.BackOff = False
                    self.LastBackOff = Time
                else:
                    self.Lambda += self.Alpha
            elif MODE[self.NodeID]<3: #honest active
                self.Lambda = NU*REP[self.NodeID]/sum(REP)
            else: # malicious
                self.Lambda = 5*NU*REP[self.NodeID]/sum(REP)
            
    def add_to_inbox(self, Packet, Time):
        """
        Add to inbox if not already received and/or processed
        """
        if Packet.Data.NodeID == self.NodeID:
            Packet.Data.IssueTime = Time
        if Packet.Data not in self.Inbox.Trans:
            if Packet.Data not in self.Ledger:
                self.Inbox.AllPackets.append(Packet)
                self.Inbox.Packets[Packet.Data.NodeID].append(Packet)
                self.Inbox.Trans.append(Packet.Data)
                self.ArrivalTimes[Packet.Data.NodeID].append(Time)
        
    
                
class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self):
        self.AllPackets = [] # Inbox_m
        self.Packets = [] # Inbox_m(i)
        self.Credits = np.zeros(NUM_NODES)
        for NodeID in range(NUM_NODES):
            self.Packets.append([])
            self.Credits[NodeID] += REP[NodeID]
        self.Trans = []
        self.Priority = np.zeros(NUM_NODES)
        self.Buffer = []
        self.RRNodeID = np.random.randint(NUM_NODES) # start at a random node
        self.Deficit = np.zeros(NUM_NODES)
        
       
    def remove_packet(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.Trans:
            if Packet in self.AllPackets:
                self.AllPackets.remove(Packet)
                self.Packets[Packet.Data.NodeID].remove(Packet)
                self.Trans.remove(Packet.Data)      
        
    def drr_schedule(self):
        if self.AllPackets:
            while True:
                if self.Deficit[self.RRNodeID]>=1 and self.Packets[self.RRNodeID]:
                    Packet = self.Packets[self.RRNodeID][0]
                    self.Deficit[self.RRNodeID] -= 1
                    # remove the transaction from all inboxes
                    self.remove_packet(Packet)
                    return Packet
                else: # move to next node and assign deficit
                    self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
                    self.Deficit[self.RRNodeID] = min(self.Deficit[self.RRNodeID]+QUANTUM*REP[self.RRNodeID], MAX_BURST*REP[self.RRNodeID]) # limited deficit savings
        else:
            self.Deficit = [MAX_BURST*REP[NodeID] for NodeID in range(NUM_NODES)]
                    
    def fifo_schedule(self, Time):
        if self.AllPackets:
            Packet = self.AllPackets[0]
            # remove the transaction from all inboxes
            self.remove_packet(Packet)
            return Packet
    
    def aimd_schedule(self, Time):
        if self.AllPackets:
            # update priority of inbox channels
            for NodeID in range(NUM_NODES):
                self.Priority[NodeID] = min(self.Priority[NodeID]+REP[NodeID], MAX_BURST*sum(REP))
            # First sort by priority
            PriorityOrder = sorted(range(NUM_NODES), key=lambda k: self.Priority[k], reverse=True)
            # take highest priority nonempty queue with oldest tx
            EarliestTime = Time
            HighestPriority = -float('Inf')
            for NodeID in PriorityOrder:
                if self.Packets[NodeID]:
                    Priority = self.Priority[NodeID]
                    if Priority>=HighestPriority:
                        HighestPriority = Priority
                        ArrivalTime = self.Packets[NodeID][0].EndTime
                        if ArrivalTime<=EarliestTime:
                            EarliestTime = ArrivalTime
                            BestNodeID = NodeID
                    else:
                        break
            Packet = self.Packets[BestNodeID][0]
            # reduce priority multiplicatively
            self.Priority[BestNodeID] = 0.5*self.Priority[BestNodeID]
            # remove the transaction from all inboxes
            self.remove_packet(Packet)
            return Packet
        
    def token_bucket_schedule(self, Time):
        if self.AllPackets:
            # update priority of inbox channels
            for NodeID in range(NUM_NODES):
                self.Priority[NodeID] = min(self.Priority[NodeID]+REP[NodeID], MAX_BURST*sum(REP))
            # First sort by priority
            PriorityOrder = sorted(range(NUM_NODES), key=lambda k: self.Priority[k], reverse=True)
            # take highest priority nonempty queue with oldest tx
            EarliestTime = Time
            HighestPriority = -float('Inf')
            for NodeID in PriorityOrder:
                if self.Packets[NodeID]:
                    Priority = self.Priority[NodeID]
                    if Priority>=HighestPriority:
                        HighestPriority = Priority
                        ArrivalTime = self.Packets[NodeID][0].EndTime
                        if ArrivalTime<=EarliestTime:
                            EarliestTime = ArrivalTime
                            BestNodeID = NodeID
                    else:
                        break
            Packet = self.Packets[BestNodeID][0]
            # reduce priority of the inbox channel by total rep amount or to zero
            self.Priority[BestNodeID] = max(self.Priority[BestNodeID]-sum(REP), 0)
            # remove the transaction from all inboxes
            self.remove_packet(Packet)
            return Packet

    def wrr_schedule(self, Time):
        if self.AllPackets:
            while True:
                if self.Packets[self.RRNodeID]:
                    Packet = self.Packets[self.RRNodeID][0]
                    self.RRSlot += 1
                    if self.RRSlot >= REP[self.RRNodeID]:
                        self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
                        self.RRSlot = 0
                    # remove the transaction from all inboxes
                    self.remove_packet(Packet)
                    return Packet
                else: # move to next node's first slot
                    self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
                    self.RRSlot = 0
                
    def brr_schedule(self, Time):
        population = [NodeID for NodeID in range(NUM_NODES) if MODE[NodeID]>0]
        while population:
            creds = [self.Credits[NodeID] for NodeID in population]
            NodeID = np.random.choice(population, p=creds/sum(creds))
            if self.Packets[NodeID]:
                self.Credits[NodeID] = REP[NodeID]
                Packet = self.Packets[NodeID][0]# remove the transaction from all inboxes
                self.remove_packet(Packet)
                return Packet
            else:
                self.Credits[NodeID] += 1
                population.remove(NodeID)

class Packet:
    """
    Object for sending data including TXs and back off notifications over
    comm channels
    """    
    def __init__(self, TxNode, RxNode, Data, StartTime):
        # can be a TX or a back off notification
        self.TxNode = TxNode
        self.RxNode = RxNode
        self.Data = Data
        self.StartTime = StartTime
        self.EndTime = []
    
class CommChannel:
    """
    Object for moving packets from node to node and simulating communication
    delays
    """
    def __init__(self, TxNode, RxNode, Delay):
        # transmitting node
        self.TxNode = TxNode
        # receiving node
        self.RxNode = RxNode
        self.Delay = Delay
        self.Packets = []
    
    def send_packet(self, TxNode, RxNode, Data, Time):
        """
        Add new packet to the comm channel with time of arrival
        """
        self.Packets.append(Packet(TxNode, RxNode, Data, Time))
    
    def transmit_packets(self, Time):
        """
        Move packets through the comm channel, simulating delay
        """
        if self.Packets:
            for Packet in self.Packets:
                if(Packet.StartTime+self.Delay<=Time):
                    self.deliver_packet(Packet, Packet.StartTime+self.Delay)
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        """
        When packet has arrived at receiving node, process it
        """
        Packet.EndTime = Time
        if isinstance(Packet.Data, Transaction):
            # if this is a transaction, add the Packet to Inbox
            self.RxNode.add_to_inbox(Packet, Time)
        else: 
            # else this is a back off notification
            self.RxNode.process_cong_notif(Packet, Time)
        self.Packets.remove(Packet)
        
class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix):
        self.A = AdjMatrix
        self.Nodes = []
        self.CommChannels = []
        self.Throughput = [0 for NodeID in range(NUM_NODES)]
        self.TranDelays = []
        self.DissemTimes = []
        Genesis = Transaction(0, [], [])
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(Node(self, i, Genesis))
        # Add neighbours and create list of comm channels corresponding to each node
        for i in range(np.size(self.A,1)):
            RowList = []
            for j in np.nditer(np.nonzero(self.A[i,:])):
                self.Nodes[i].Neighbours.append(self.Nodes[j])
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.A[i][j]))
            self.CommChannels.append(RowList)
            
    def send_data(self, TxNode, RxNode, Data, Time):
        """
        Send this data (TX or back off) to a specified neighbour
        """
        CC = self.CommChannels[TxNode.NodeID][TxNode.Neighbours.index(RxNode)]
        CC.send_packet(TxNode, RxNode, Data, Time)
        
    def broadcast_data(self, TxNode, LastTxNode, Data, Time):
        """
        Send this data (TX or back off) to all neighbours
        """
        for i, CC in enumerate(self.CommChannels[self.Nodes.index(TxNode)]):
            # do not send to this node if it was received from this node
            if isinstance(Data, Transaction):
                if LastTxNode==TxNode.Neighbours[i]:
                    continue
            CC.send_packet(TxNode, TxNode.Neighbours[i], Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate and process new transactions
        """
        for Node in self.Nodes:
            Node.generate_txs(Time)
            Node.schedule_txs(Time)
            Node.check_congestion(Time)
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time+STEP)
    
    def sym_diffs(self):
        SymDiffs = np.zeros((NUM_NODES, NUM_NODES))
        for i, iNode in enumerate(self.Nodes):
            for j, jNode in enumerate(self.Nodes):
                if j>i:
                    SymDiffs[i][j] = len(set(iNode.Ledger).symmetric_difference(set(jNode.Ledger)))
        return SymDiffs + np.transpose(SymDiffs)
    
    def tran_latency(self, latencies, latTimes):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.IssueTime>20:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.IssueTime)
                latTimes[Tran.NodeID].append(Tran.GlobalSolidTime)
        return latencies, latTimes
                
if __name__ == "__main__":
        main()
        
    
    