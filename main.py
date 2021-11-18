# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import os
import sys
from time import gmtime, strftime
from core.global_params import *
from core.network import Network
from utils import all_node_plot, per_node_barplot, per_node_plot, plot_cdf


np.random.seed(0)
    
def main():
    '''
    Create directory for storing results with these parameters
    '''
    dirstr = simulate()
    plot_results(dirstr)
    
    sys.stdout.write('\a')
    
def simulate():
    """
    Setup simulation inputs and instantiate output arrays
    """
    # seed rng
    np.random.seed(0)
    TimeSteps = int(SIM_TIME/STEP)
    
    """
    Monte Carlo Sims
    """
    Lmds = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    OldestTxAges = np.zeros((TimeSteps, NUM_NODES))
    OldestTxAge = []
    InboxLens = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    ReadyLens = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    NumTips = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    InboxLensMA = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Deficits = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Throughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    WorkThroughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Undissem = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    MeanDelay = [np.zeros(SIM_TIME) for mc in range(MONTE_CARLOS)]
    MeanVisDelay = [np.zeros(SIM_TIME) for mc in range(MONTE_CARLOS)]
    Unsolid = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    EligibleDelays = [np.zeros((SIM_TIME, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    TP = []
    WTP = []
    latencies = [[] for NodeID in range(NUM_NODES)]
    inboxLatencies = [[] for NodeID in range(NUM_NODES)]
    latTimes = [[] for NodeID in range(NUM_NODES)]
    ServTimes = [[] for NodeID in range(NUM_NODES)]
    ArrTimes = [[] for NodeID in range(NUM_NODES)]
    interArrTimes = [[] for NodeID in range(NUM_NODES)]
    for mc in range(MONTE_CARLOS):
        """
        Generate network topology:
        Comment out one of the below lines for either random k-regular graph or a
        graph from an adjlist txt file i.e. from the autopeering simulator
        """
        if GRAPH=='regular':
            G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES) # random regular graph
        elif GRAPH=='complete':
            G = nx.complete_graph(NUM_NODES) # complete graph
        elif GRAPH=='cycle':
            G = nx.cycle_graph(NUM_NODES) # cycle graph
        #G = nx.read_adjlist('input_adjlist.txt', delimiter=' ')
        # Get adjacency matrix and weight by delay at each channel
        ChannelDelays = 0.05*np.ones((NUM_NODES, NUM_NODES))+0.1*np.random.rand(NUM_NODES, NUM_NODES)
        AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
        Net = Network(AdjMatrix)
        # output arrays
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
                if Net.Nodes[NodeID].Inbox.AllPackets and MODE[NodeID]<3: #don't include malicious nodes
                    HonestPackets = [p for p in Net.Nodes[NodeID].Inbox.AllPackets if MODE[p.Data.NodeID]<3]
                    if HonestPackets:
                        OldestPacket = min(HonestPackets, key=lambda x: x.Data.IssueTime)
                        OldestTxAges[i,NodeID] = T - OldestPacket.Data.IssueTime
                InboxLens[mc][i,NodeID] = len(Net.Nodes[NodeID].Inbox.AllPackets)
                ReadyLens[mc][i,NodeID] = len(Net.Nodes[NodeID].Inbox.ReadyPackets)
                NumTips[mc][i,NodeID] = len(Net.Nodes[NodeID].TipsSet)
                Unsolid[mc][i,NodeID] = len([tran for tran in Net.Nodes[NodeID].Ledger if not tran.Solid])
                InboxLensMA[mc][i,NodeID] = Net.Nodes[NodeID].Inbox.Avg
                Deficits[mc][i, NodeID] = Net.Nodes[0].Inbox.Deficit[NodeID]
                Throughput[mc][i, NodeID] = Net.Throughput[NodeID]
                WorkThroughput[mc][i,NodeID] = Net.WorkThroughput[NodeID]
                Undissem[mc][i,NodeID] = Net.Nodes[NodeID].Undissem
        print("Simulation: "+str(mc) +"\t 100% Complete")
        OldestTxAge.append(np.mean(OldestTxAges, axis=1))
        for i in range(SIM_TIME):
            delays = [Net.TranDelays[j] for j in Net.TranDelays.keys() if int(Net.DissemTimes[j])==i]
            if delays:
                MeanDelay[mc][i] = sum(delays)/len(delays)
            visDelays = [Net.VisTranDelays[j] for j in Net.VisTranDelays.keys() if int(Net.DissemTimes[j])==i]
            if visDelays:
                MeanVisDelay[mc][i] = sum(visDelays)/len(visDelays)
        for NodeID in range(NUM_NODES):
            for i in range(SIM_TIME):
                delays = []
                for tran in Net.Nodes[NodeID].Ledger:
                    if tran.EligibleTime is not None:
                        if int(tran.EligibleTime)==i:
                            delays.append(tran.EligibleTime-tran.IssueTime)
                if delays:
                    EligibleDelays[mc][i, NodeID] = sum(delays)/len(delays)
                
            ServTimes[NodeID] = sorted(Net.Nodes[NodeID].ServiceTimes)
            ArrTimes[NodeID] = sorted(Net.Nodes[NodeID].ArrivalTimes)
            ArrWorks = [x for _,x in sorted(zip(Net.Nodes[NodeID].ArrivalTimes,Net.Nodes[NodeID].ArrivalWorks))]
            interArrTimes[NodeID].extend(np.diff(ArrTimes[NodeID])/ArrWorks[1:])
            inboxLatencies[NodeID].extend(Net.Nodes[NodeID].InboxLatencies)
                
        latencies, latTimes = Net.tran_latency(latencies, latTimes)
        
        TP.append(np.concatenate((np.zeros((1000, NUM_NODES)),(Throughput[mc][1000:,:]-Throughput[mc][:-1000,:])))/10)
        WTP.append(np.concatenate((np.zeros((1000, NUM_NODES)),(WorkThroughput[mc][1000:,:]-WorkThroughput[mc][:-1000,:])))/10)
        #TP.append(np.convolve(np.zeros((Throughput[mc][500:,:]-Throughput[mc][:-500,:])))/5)
        del Net
    """
    Get results
    """
    avgLmds = sum(Lmds)/len(Lmds)
    avgTP = sum(TP)/len(TP)
    avgWTP = sum(WTP)/len(WTP)
    avgInboxLen = sum(InboxLens)/len(InboxLens)
    avgReadyLen = sum(ReadyLens)/len(ReadyLens)
    avgNumTips = sum(NumTips)/len(NumTips)
    avgInboxLenMA = sum(InboxLensMA)/len(InboxLensMA)
    avgDefs = sum(Deficits)/len(Deficits)
    avgUndissem = sum(Undissem)/len(Undissem)
    avgMeanDelay = sum(MeanDelay)/len(MeanDelay)
    avgMeanVisDelay = sum(MeanVisDelay)/len(MeanVisDelay)
    avgUnsolid = sum(Unsolid)/len(Unsolid)
    avgEligibleDelays = sum(EligibleDelays)/len(EligibleDelays)
    avgOTA = sum(OldestTxAge)/len(OldestTxAge)
    """
    Create a directory for these results and save them
    """
    dirstr = os.path.dirname(os.path.realpath(__file__)) + '/results/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    os.makedirs(dirstr+'/raw', exist_ok=True)
    os.makedirs(dirstr+'/plots', exist_ok=True)
    np.savetxt(dirstr+'/config.txt', ['MCs = ' + str(MONTE_CARLOS) +
                                      '\nsimtime = ' + str(SIM_TIME) +
                                      '\nstep = ' + str(STEP) +
                                      '\n\n# Network Parameters' +
                                      '\nnu = ' + str(NU) +
                                      '\nnumber of nodes = ' + str(NUM_NODES) +
                                      '\nnumber of neighbours = ' + str(NUM_NEIGHBOURS) +
                                      '\ngraph topology = ' + GRAPH +
                                      '\nrepdist = ' + str(REPDIST) +
                                      '\nmodes = ' + str(MODE) +
                                      '\niot = ' + str(IOT) +
                                      '\niotlow = ' + str(IOTLOW) +
                                      '\niothigh = ' + str(IOTHIGH) +
                                      '\ndcmax = ' + str(MAX_WORK) +
                                      '\n\n# Congestion Control Parameters' +
                                      '\nalpha = ' + str(ALPHA) +
                                      '\nbeta = ' + str(BETA) +
                                      '\ntau = ' + str(TAU) + 
                                      '\nminth = ' + str(MIN_TH) +
                                      '\nmaxth = ' + str(MAX_TH) +
                                      '\nquantum = ' + str(QUANTUM) +
                                      '\nw_q = ' + str(W_Q) +
                                      '\np_b = ' + str(P_B) +
                                      '\nsched=' + SCHEDULING], delimiter = " ", fmt='%s')
    
    np.savetxt(dirstr+'/raw/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/raw/avgTP.csv', avgTP, delimiter=',')
    np.savetxt(dirstr+'/raw/avgWTP.csv', avgWTP, delimiter=',')
    np.savetxt(dirstr+'/raw/avgInboxLen.csv', avgInboxLen, delimiter=',')
    np.savetxt(dirstr+'/raw/avgReadyLen.csv', avgReadyLen, delimiter=',')
    np.savetxt(dirstr+'/raw/avgNumTips.csv', avgNumTips, delimiter=',')
    np.savetxt(dirstr+'/raw/avgInboxLenMA.csv', avgInboxLenMA, delimiter=',')
    np.savetxt(dirstr+'/raw/avgDefs.csv', avgDefs, delimiter=',')
    np.savetxt(dirstr+'/raw/avgUndissem.csv', avgUndissem, delimiter=',')
    np.savetxt(dirstr+'/raw/avgMeanDelay.csv', avgMeanDelay, delimiter=',')
    np.savetxt(dirstr+'/raw/avgMeanVisDelay.csv', avgMeanVisDelay, delimiter=',')
    np.savetxt(dirstr+'/raw/avgOldestTxAge.csv', avgOTA, delimiter=',')
    np.savetxt(dirstr+'/raw/avgUnsolid.csv', avgUnsolid, delimiter=',')
    np.savetxt(dirstr+'/raw/avgEligibleDelays.csv', avgEligibleDelays, delimiter=',')
    for NodeID in range(NUM_NODES):
        np.savetxt(dirstr+'/raw/inboxLatencies'+str(NodeID)+'.csv',
                   np.asarray(inboxLatencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/latencies'+str(NodeID)+'.csv',
                   np.asarray(latencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/ServTimes'+str(NodeID)+'.csv',
                   np.asarray(ServTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/ArrTimes'+str(NodeID)+'.csv',
                   np.asarray(ArrTimes[NodeID]), delimiter=',')
    nx.write_adjlist(G, dirstr+'/raw/result_adjlist.txt', delimiter=' ')
    return dirstr


    
def plot_results(dirstr):
    """
    Initialise plots
    """
    plt.close('all')
    
    """
    Load results from the data directory
    """
    avgLmds = np.loadtxt(dirstr+'/raw/avgLmds.csv', delimiter=',')
    #avgTP = np.loadtxt(dirstr+'/avgTP.csv', delimiter=',')
    avgTP = np.loadtxt(dirstr+'/raw/avgWTP.csv', delimiter=',')
    avgInboxLen = np.loadtxt(dirstr+'/raw/avgInboxLen.csv', delimiter=',')
    avgReadyLen = np.loadtxt(dirstr+'/raw/avgReadyLen.csv', delimiter=',')
    avgNumTips = np.loadtxt(dirstr+'/raw/avgNumTips.csv', delimiter=',')
    avgUnsolid = np.loadtxt(dirstr+'/raw/avgUnsolid.csv', delimiter=',')
    avgEligibleDelays = np.loadtxt(dirstr+'/raw/avgEligibleDelays.csv', delimiter=',')
    avgInboxLenMA = np.loadtxt(dirstr+'/raw/avgInboxLenMA.csv', delimiter=',')
    avgUndissem = np.loadtxt(dirstr+'/raw/avgUndissem.csv', delimiter=',')
    avgMeanDelay = np.loadtxt(dirstr+'/raw/avgMeanDelay.csv', delimiter=',')
    #avgMeanDelay = np.loadtxt(dirstr+'/avgMeanVisDelay.csv', delimiter=',')
    avgOTA = np.loadtxt(dirstr+'/raw/avgOldestTxAge.csv', delimiter=',')
    latencies = []
    #inboxLatencies = []
    ServTimes = []
    ArrTimes = []
    
    for NodeID in range(NUM_NODES):
        if os.stat(dirstr+'/raw/latencies'+str(NodeID)+'.csv').st_size != 0:
            lat = [np.loadtxt(dirstr+'/raw/latencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            lat = [0]
        latencies.append(lat)
        '''
        if os.stat(dirstr+'/InboxLatencies'+str(NodeID)+'.csv').st_size != 0:
            inbLat = [np.loadtxt(dirstr+'/inboxLatencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            inbLat = [0]
        inboxLatencies.append(inbLat)
        '''
        ServTimes.append([np.loadtxt(dirstr+'/raw/ServTimes'+str(NodeID)+'.csv', delimiter=',')])
        ArrTimes.append([np.loadtxt(dirstr+'/raw/ArrTimes'+str(NodeID)+'.csv', delimiter=',')])
    """
    Plot results
    """
    fig1, ax1 = plt.subplots(2,1, sharex=True, figsize=(8,8))
    ax1[0].title.set_text('Dissemination Rate')
    ax1[1].title.set_text('Scaled Dissemination Rate')
    ax1[0].grid(linestyle='--')
    ax1[1].grid(linestyle='--')
    ax1[1].set_xlabel('Time (sec)')
    #ax1[0].set_ylabel(r'${\lambda_i} / {\~{\lambda}_i}$')
    ax1[0].set_ylabel(r'$D_i$')
    ax1[1].set_ylabel(r'$D_i / {\~{\lambda}_i}$')
    mal = False
    iot = False
    for NodeID in range(NUM_NODES):
        if IOT[NodeID]:
            iot = True
            marker = 'x'
        else:
            marker = None
        if MODE[NodeID]==1:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:blue', marker=marker, markevery=0.1)
        if MODE[NodeID]==2:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:red', marker=marker, markevery=0.1)
        if MODE[NodeID]==3:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:green', marker=marker, markevery=0.1)
            mal = True
    if mal:
        ModeLines = [Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='tab:green', lw=4)]
        fig1.legend(ModeLines, ['Content','Best-effort', 'Malicious'], loc='right')
    elif iot:
        ModeLines = [Line2D([0],[0],color='tab:blue'), Line2D([0],[0],color='tab:red'), Line2D([0],[0],color='tab:blue', marker='x'), Line2D([0],[0],color='tab:red', marker='x')]
        fig1.legend(ModeLines, ['Content value node','Best-effort value node', 'Content IoT node', 'Best-effort IoT node'], loc='right')
    else:
        ModeLines = [Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4)]
        fig1.legend(ModeLines, ['Content','Best-effort'], loc='right')
    plt.savefig(dirstr+'/plots/Rates.png', bbox_inches='tight')
    
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Time (sec)')
    ax2.plot(np.arange(10, SIM_TIME, STEP), 100*np.sum(avgTP[1000:,:], axis=1)/NU, color = 'black')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay, color='tab:gray')    
    ax2.tick_params(axis='y', labelcolor='black')
    ax22.tick_params(axis='y', labelcolor='tab:gray')
    ax2.set_ylabel(r'$DR/\nu \quad (\%)$', color='black')
    ax2.set_ylim([0,110])
    ax22.set_ylabel('Mean Latency (sec)', color='tab:gray')
    #ax22.set_ylim([0,2])
    fig2.tight_layout()
    plt.savefig(dirstr+'/plots/Throughput.png', bbox_inches='tight')
    

    plot_cdf(latencies, 'Latency (sec)', dirstr+'/plots/Latency.png')
    
    #ax4.plot(np.arange(0, SIM_TIME, STEP), np.sum(avgLmds, axis=1), color='tab:blue')

    per_node_plot(avgLmds, 'Time (sec)', r'$\lambda_i$', '', dirstr+'/plots/IssueRates.png', avg_window=1, modes=[1,2])
    
    per_node_plot(avgInboxLen, 'Time (sec)', 'Inbox length', '', dirstr+'/plots/AvgInboxLen.png')
    per_node_plot(avgReadyLen, 'Time (sec)', 'Ready length', '', dirstr+'/plots/AvgReadyLen.png')
    per_node_plot(avgNumTips, 'Time (sec)', 'Number of Tips', '', dirstr+'/plots/AvgNumTips.png')
    per_node_plot(avgEligibleDelays, 'Time (sec)', 'Eligible Delays', '', dirstr+'/plots/AvgEligibleDelays.png', avg_window=20, step=1)
    per_node_plot(avgUnsolid, 'Time (sec)', 'Unsolid', '', dirstr+'/plots/AvgUnsolid.png')
    
    fig5a, ax5a = plt.subplots(figsize=(8,4))
    ax5a.grid(linestyle='--')
    ax5a.set_xlabel('Time (sec)')
    ax5a.set_ylabel('Inbox Len and Arrivals')
    NodeID = 2
    step = 1
    bins = np.arange(0, SIM_TIME, step)
    i = 0
    j = 0
    nArr = np.zeros(len(bins))
    inboxLen = np.zeros(len(bins))
    for b, t in enumerate(bins):
        if b>0:
            inboxLen[b] = inboxLen[b-1]
        while ArrTimes[NodeID][0][i] < t+step:
            nArr[b] += 1
            inboxLen[b] +=1
            i += 1
            if i>=len(ArrTimes[NodeID][0]):
                break
            
        while ServTimes[NodeID][0][j] < t+step:
            inboxLen[b] -= 1
            j += 1
            if j>=len(ServTimes[NodeID][0]):
                break
    ax5a.plot(bins, nArr/(step*NU), color = 'black')
    ax5b = ax5a.twinx()
    ax5b.plot(bins, inboxLen, color='blue')
    
    plt.savefig(dirstr+'/plots/InboxLenMA.png', bbox_inches='tight')
    
    per_node_barplot('Node ID', 'Reputation', 'Reputation Distribution', dirstr+'/plots/RepDist.png')

    all_node_plot(avgOTA, 'Time (sec)', 'Max time in transit (sec)', '', dirstr+'/plots/MaxAge.png')

if __name__ == "__main__":
        main()