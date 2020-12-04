# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from random import sample
import os
import winsound
from time import gmtime, strftime
from shutil import copyfile
import csv

np.random.seed(0)
# Simulation Parameters
MONTE_CARLOS = 1
SIM_TIME = 180
STEP = 0.01
# Network Parameters
GRAPH = 'cycle' # regular, complete or cycle
NU = 50
NUM_NODES = 50
NUM_NEIGHBOURS = 4
CHAN_DELAY_MEAN = 0.05
START_TIMES = 10*np.ones(NUM_NODES)
REPDIST = 'zipf'
if REPDIST=='zipf':
    # IOTA data rep distribution - Zipf s=0.9
    REP = [(NUM_NODES+1)/((NodeID+1)**0.9) for NodeID in range(NUM_NODES)]
elif REPDIST=='uniform':
    # Permissioned System rep system?
    REP = np.ones(NUM_NODES, dtype=int)
# Modes: 0 = inactive, 1 = content, 2 = best-effort, 3 = malicious
MODE = [2-NodeID%3 for NodeID in range(NUM_NODES)] # honest env
#MODE = [3-(NodeID+1)%4 for NodeID in range(NUM_NODES)] # malicoius env
#MODE[2]=4
#AVG_WORK = [2*rep/max(REP) for rep in REP
#AVG_WORK = np.ones(NUM_NODES)
#IOT = np.concatenate((np.zeros(int(NUM_NODES/2)), np.ones(int(NUM_NODES/2))))
#IOT = [NodeID%2 for NodeID in range(NUM_NODES)]
IOT = np.zeros(NUM_NODES)
IOTLOW = 0.5
IOTHIGH = 1
MAX_WORK = 10
MODE_COLOUR_MAP = ['grey', 'tab:blue', 'tab:red', 'tab:green', 'tab:green']

# Congestion Control Parameters
ALPHA = 0.075
BETA = 0.5
TAU = 2
MIN_TH = 1
MAX_TH = MIN_TH
QUANTUM = [MAX_WORK*rep/sum(REP) for rep in REP]
W_Q = 0.1
P_B = 0.5
DROP_TH = 5
SCHEDULE_ON_SOLID = True
SOLID_REQUESTS = True
BLACKLIST = True
SCHEDULING = 'drr_lds'
    
def main():

    plt.close('all')
    dirstr = simulate()
    #dirstr = 'data/2020-08-19_142643'
    plot_results(dirstr)
    #plot_results('data/2020-07-23_213333') # honest environment
    #plot_results('data/2020-07-26_122609') # malicious environment
    #plot_results('data/2020-07-25_154015') # IoT
    #plot_scheduler_comp('Data/2020-07-23_233549', 'Data/2020-07-23_213333')
    #plot_ratesetter_comp('data/2020-07-24_132945', 'data/2020-07-23_213333', 'data/2020-07-24_153139') # changing beta
    #plot_ratesetter_comp('data/2020-07-23_212831', 'data/2020-07-23_213333', 'data/2020-07-24_124532') # changing W
    #plot_ratesetter_comp('data/2020-07-23_213056', 'data/2020-07-23_213333', 'data/2020-07-24_154211') # changing alpha
    winsound.Beep(2500, 1000) # beep to say sim is finished
    
def simulate():
    """
    Setup simulation inputs and instantiate output arrays
    """
    # seed rng
    np.random.seed(0)
    TimeSteps = int(SIM_TIME/STEP)
    '''
    Create empty arrays to store results
    '''
    Lmds = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    OldestTxAges = np.zeros((TimeSteps, NUM_NODES))
    OldestTxAge = []
    InboxLens = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    InboxLensMA = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    SolidRequests = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Deficits = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Throughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    WorkThroughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Undissem = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    MeanDelay = [np.zeros(SIM_TIME) for mc in range(MONTE_CARLOS)]
    MeanVisDelay = [np.zeros(SIM_TIME) for mc in range(MONTE_CARLOS)]
    TP = []
    WTP = []
    latencies = [[] for NodeID in range(NUM_NODES)]
    inboxLatencies = [[] for NodeID in range(NUM_NODES)]
    latTimes = [[] for NodeID in range(NUM_NODES)]
    ServTimes = [[] for NodeID in range(NUM_NODES)]
    ArrTimes = [[] for NodeID in range(NUM_NODES)]
    interArrTimes = [[] for NodeID in range(NUM_NODES)]
    DroppedTrans = [[] for NodeID in range(NUM_NODES)]
    DropTimes = [[] for NodeID in range(NUM_NODES)]
    """
    Monte Carlo Sims
    """
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
        ChannelDelays = 0.05*np.ones((NUM_NODES, NUM_NODES))+0.1*np.random.rand(NUM_NODES, NUM_NODES) # not used anymore
        AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
        Net = Network(AdjMatrix) # instantiate the network
        
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
            '''
            save summary results in output arrays
            '''
            for NodeID in range(NUM_NODES):
                Lmds[mc][i, NodeID] = Net.Nodes[NodeID].Lambda
                if Net.Nodes[NodeID].Inbox.AllPackets and MODE[NodeID]<3: #don't include malicious nodes
                    HonestPackets = [p for p in Net.Nodes[NodeID].Inbox.AllPackets if MODE[p.Data.NodeID]<3]
                    if HonestPackets:
                        OldestPacket = min(HonestPackets, key=lambda x: x.Data.IssueTime)
                        OldestTxAges[i,NodeID] = T - OldestPacket.Data.IssueTime
                InboxLens[mc][i, NodeID] = len(Net.Nodes[2].Neighbours[0].Inbox.Packets[NodeID])/REP[NodeID]
                InboxLensMA[mc][i,NodeID] = len(Net.Nodes[2].Neighbours[0].Neighbours[-1].Inbox.Packets[NodeID])/REP[NodeID]
                SolidRequests[mc][i,NodeID] = len(Net.Nodes[NodeID].Inbox.RequestedTrans)
                Deficits[mc][i, NodeID] = Net.Nodes[6].Inbox.Deficit[NodeID]
                Throughput[mc][i, NodeID] = Net.Throughput[NodeID]
                WorkThroughput[mc][i,NodeID] = Net.WorkThroughput[NodeID]
                Undissem[mc][i,NodeID] = Net.Nodes[NodeID].Undissem
        print("Simulation: "+str(mc) +"\t 100% Complete")
        OldestTxAge.append(np.mean(OldestTxAges, axis=1))
        for NodeID in range(NUM_NODES):
            for i in range(SIM_TIME):
                delays = [Net.TranDelays[j] for j in range(len(Net.TranDelays)) if int(Net.DissemTimes[j])==i]
                if delays:
                    MeanDelay[mc][i] = sum(delays)/len(delays)
                visDelays = [Net.VisTranDelays[j] for j in range(len(Net.VisTranDelays)) if int(Net.DissemTimes[j])==i]
                if visDelays:
                    MeanVisDelay[mc][i] = sum(visDelays)/len(visDelays)
                
            ServTimes[NodeID] = sorted(Net.Nodes[NodeID].ServiceTimes)
            ArrTimes[NodeID] = sorted(Net.Nodes[NodeID].ArrivalTimes)
            ArrWorks = [x for _,x in sorted(zip(Net.Nodes[NodeID].ArrivalTimes,Net.Nodes[NodeID].ArrivalWorks))]
            interArrTimes[NodeID].extend(np.diff(ArrTimes[NodeID])/ArrWorks[1:])
            inboxLatencies[NodeID].extend(Net.Nodes[NodeID].InboxLatencies)
            DroppedTrans[NodeID].extend(Net.Nodes[NodeID].Inbox.DroppedTrans)
            DropTimes[NodeID].extend(Net.Nodes[NodeID].Inbox.DropTimes)
            
        Blacklist = [Net.Nodes[NodeID].Blacklist for NodeID in range(NUM_NODES)]
        latencies, latTimes = Net.tran_latency(latencies, latTimes)
        
        window = 50
        TP.append(np.concatenate((np.zeros((int(window/STEP), NUM_NODES)),(Throughput[mc][int(window/STEP):,:]-Throughput[mc][:-int(window/STEP),:])))/window)
        WTP.append(np.concatenate((np.zeros((int(window/STEP), NUM_NODES)),(WorkThroughput[mc][int(window/STEP):,:]-WorkThroughput[mc][:-int(window/STEP),:])))/window)
        #TP.append(np.convolve(np.zeros((Throughput[mc][500:,:]-Throughput[mc][:-500,:])))/5)
        Neighbours = [[Neighb.NodeID for Neighb in Node.Neighbours] for Node in Net.Nodes]
        del Net
    """
    Get results
    """
    avgLmds = sum(Lmds)/len(Lmds)
    avgTP = sum(TP)/len(TP)
    avgWTP = sum(WTP)/len(WTP)
    avgInboxLen = sum(InboxLens)/len(InboxLens)
    avgInboxLenMA = sum(InboxLensMA)/len(InboxLensMA)
    avgSolReq = sum(SolidRequests)/len(SolidRequests)
    avgDefs = sum(Deficits)/len(Deficits)
    avgUndissem = sum(Undissem)/len(Undissem)
    avgMeanDelay = sum(MeanDelay)/len(MeanDelay)
    avgMeanVisDelay = sum(MeanVisDelay)/len(MeanVisDelay)
    avgOTA = sum(OldestTxAge)/len(OldestTxAge)
    """
    Create a directory for these results and save them
    """
    dirstr = 'data/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,
                   nodelist=range(NUM_NODES),
                   node_color=[MODE_COLOUR_MAP[MODE[NodeID]] for NodeID in range(NUM_NODES)],
                   node_size=200,
               alpha=0.8)
    nx.draw_networkx_edges(G,pos,width=1.0,alpha=0.5)
    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    plt.axis('off')
    plt.savefig(dirstr+'/Graph.png', bbox_inches='tight')
    np.savetxt(dirstr+'/aaconfig.txt', ['MCs = ' + str(MONTE_CARLOS) +
                                      '\nsimtime = ' + str(SIM_TIME) +
                                      '\nstep = ' + str(STEP) +
                                      '\n\n# Network Parameters' +
                                      '\nnu = ' + str(NU) +
                                      '\ngraph type = ' + GRAPH +
                                      '\nnumber of nodes = ' + str(NUM_NODES) +
                                      '\nnumber of neighbours = ' + str(NUM_NEIGHBOURS) +
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
                                      '\nSchedule on solid = ' + str(SCHEDULE_ON_SOLID) +
                                      '\nsolidification requests = ' + str(SOLID_REQUESTS) +
                                      '\nblacklist = ' + str(BLACKLIST) +
                                      '\nsched=' + SCHEDULING], delimiter = " ", fmt='%s')
    
    np.savetxt(dirstr+'/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/avgTP.csv', avgTP, delimiter=',')
    np.savetxt(dirstr+'/avgWTP.csv', avgWTP, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLen.csv', avgInboxLen, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLenMA.csv', avgInboxLenMA, delimiter=',')
    np.savetxt(dirstr+'/avgSolReq.csv', avgSolReq, delimiter=',')
    np.savetxt(dirstr+'/avgDefs.csv', avgDefs, delimiter=',')
    np.savetxt(dirstr+'/avgUndissem.csv', avgUndissem, delimiter=',')
    np.savetxt(dirstr+'/avgMeanDelay.csv', avgMeanDelay, delimiter=',')
    np.savetxt(dirstr+'/avgMeanVisDelay.csv', avgMeanVisDelay, delimiter=',')
    np.savetxt(dirstr+'/avgOldestTxAge.csv', avgOTA, delimiter=',')
    #np.savetxt(dirstr+'/adjlist.csv', np.asarray(Neighbours), delimiter=',')
    f = open(dirstr+'/blacklist.csv','w')
    with f:
        writer = csv.writer(f)
        for row in Blacklist:
            if row:
                writer.writerow(row)
            else:
                writer.writerow('None')
    for NodeID in range(NUM_NODES):
        np.savetxt(dirstr+'/latencies'+str(NodeID)+'.csv',
                   np.asarray(latencies[NodeID]), delimiter=',')
        '''
        np.savetxt(dirstr+'/inboxLatencies'+str(NodeID)+'.csv',
                   np.asarray(inboxLatencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/ServTimes'+str(NodeID)+'.csv',
                   np.asarray(ServTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/ArrTimes'+str(NodeID)+'.csv',
                   np.asarray(ArrTimes[NodeID]), delimiter=',')
        '''
        if DroppedTrans[NodeID]:
            Drops = np.zeros((len(DroppedTrans[NodeID]), 3))
            for i in range(len(DroppedTrans[NodeID])):
                Drops[i, 0] = DroppedTrans[NodeID][i].NodeID
                Drops[i, 1] = DroppedTrans[NodeID][i].Index
                Drops[i, 2] = DropTimes[NodeID][i]
            np.savetxt(dirstr+'/Drops'+str(NodeID)+'.csv', Drops, delimiter=',')
        
    return dirstr

def plot_ratesetter_comp(dir1, dir2, dir3):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.grid(linestyle='--')
    ax.set_xlabel('Time (sec)')
    axt = ax.twinx()  
    ax.tick_params(axis='y', labelcolor='black')
    axt.tick_params(axis='y', labelcolor='tab:gray')
    ax.set_ylabel('Dissemination rate (Work/sec)', color='black')
    axt.set_ylabel('Mean Latency (sec)', color='tab:gray')
    
    avgTP1 = np.loadtxt(dir1+'/avgTP.csv', delimiter=',')
    avgMeanDelay1 = np.loadtxt(dir1+'/avgMeanDelay.csv', delimiter=',')
    avgTP2 = np.loadtxt(dir2+'/avgTP.csv', delimiter=',')
    avgMeanDelay2 = np.loadtxt(dir2+'/avgMeanDelay.csv', delimiter=',')
    avgTP3 = np.loadtxt(dir3+'/avgTP.csv', delimiter=',')
    avgMeanDelay3 = np.loadtxt(dir3+'/avgMeanDelay.csv', delimiter=',')
    
    markerevery = 200
    ax.plot(np.arange(10, SIM_TIME, STEP), np.sum(avgTP1[1000:,:], axis=1), color = 'black', marker = 'o', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay1, color='tab:gray', marker = 'o', markevery=int(markerevery*STEP)) 
    ax.plot(np.arange(10, SIM_TIME, STEP), np.sum(avgTP2[1000:,:], axis=1), color = 'black', marker = 'x', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay2, color='tab:gray', marker = 'x', markevery=int(markerevery*STEP))  
    ax.plot(np.arange(10, SIM_TIME, STEP), np.sum(avgTP3[1000:,:], axis=1), color = 'black', marker = '^', markevery=markerevery)
    axt.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay3, color='tab:gray', marker = '^', markevery=int(markerevery*STEP))  
    
    ModeLines = [Line2D([0],[0],color='black', linestyle=None, marker='o'), Line2D([0],[0],color='black', linestyle=None, marker='x'), Line2D([0],[0],color='black', linestyle=None, marker='^')]
    #ax.legend(ModeLines, [r'$A=0.05$', r'$A=0.075$', r'$A=0.1$'], loc='lower right')
    #ax.set_title(r'$\beta=0.7, \quad W=2$')
    ax.legend(ModeLines, [r'$\beta=0.5$', r'$\beta=0.7$', r'$\beta=0.9$'], loc='lower right')
    ax.set_title(r'$A=0.075, \quad W=2$')
    #ax.legend(ModeLines, [r'$W=1$', r'$W=2$', r'$W=3$'], loc='lower right')
    #ax.set_title(r'$A=0.075, \quad \beta=0.7$')
    
    dirstr = 'data/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
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
    
    dirstr = 'data/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    
    copyfile(dir1+'/aaconfig.txt', dirstr+'/config1.txt')
    copyfile(dir2+'/aaconfig.txt', dirstr+'/config2.txt')
    plt.savefig(dirstr+'/LatencyComp.png', bbox_inches='tight')
    
def plot_results(dirstr):
    """
    Initialise plots
    """
    plt.close('all')
    
    """
    Load results from the data directory
    """
    avgLmds = np.loadtxt(dirstr+'/avgLmds.csv', delimiter=',')
    #avgTP = np.loadtxt(dirstr+'/avgTP.csv', delimiter=',')
    avgTP = np.loadtxt(dirstr+'/avgWTP.csv', delimiter=',')
    avgInboxLen = np.loadtxt(dirstr+'/avgInboxLen.csv', delimiter=',')
    avgInboxLenMA = np.loadtxt(dirstr+'/avgInboxLenMA.csv', delimiter=',')
    avgSolReq = np.loadtxt(dirstr+'/avgSolReq.csv', delimiter=',')
    avgUndissem = np.loadtxt(dirstr+'/avgUndissem.csv', delimiter=',')
    avgMeanDelay = np.loadtxt(dirstr+'/avgMeanDelay.csv', delimiter=',')
    avgOTA = np.loadtxt(dirstr+'/avgOldestTxAge.csv', delimiter=',')
    avgDefs = np.loadtxt(dirstr+'/avgDefs.csv', delimiter=',')
    latencies = []
    ServTimes = []
    ArrTimes = []
    
    for NodeID in range(NUM_NODES):
        if os.stat(dirstr+'/latencies'+str(NodeID)+'.csv').st_size != 0:
            lat = [np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
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
        #ServTimes.append([np.loadtxt(dirstr+'/ServTimes'+str(NodeID)+'.csv', delimiter=',')])
        #ArrTimes.append([np.loadtxt(dirstr+'/ArrTimes'+str(NodeID)+'.csv', delimiter=',')])
    """
    Plot results
    """       
    window = 50
    fig1, ax1 = plt.subplots(2,1, sharex=True, figsize=(8,8))
    ax1[0].title.set_text('Dissemination Rate')
    ax1[1].title.set_text('Scaled Dissemination Rate')
    ax1[0].grid(linestyle='--')
    ax1[1].grid(linestyle='--')
    ax1[1].set_xlabel('Time (sec)')
    #ax1[0].set_ylabel(r'${\lambda_i} / {\~{\lambda}_i}$')
    ax1[0].set_ylabel(r'$DR_i$')
    ax1[1].set_ylabel(r'$DR_i / {\~{\lambda}_i}$')
    mal = False
    iot = False
    for NodeID in range(NUM_NODES):
        if IOT[NodeID]:
            iot = True
            marker = 'x'
        else:
            marker = None
        if MODE[NodeID]==1:
            ax1[0].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:blue', marker=marker, markevery=0.1)
        if MODE[NodeID]==2:
            ax1[0].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:red', marker=marker, markevery=0.1)
        if MODE[NodeID]>2:
            ax1[0].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green', marker=marker, markevery=0.1)
            ax1[1].plot(np.arange(window, SIM_TIME, STEP), avgTP[int(window/STEP):,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:green', marker=marker, markevery=0.1)
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
    plt.savefig(dirstr+'/Rates.png', bbox_inches='tight')
    
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Time (sec)')
    ax2.plot(np.arange(window, SIM_TIME, STEP), np.sum(avgTP[int(window/STEP):,:], axis=1), color = 'black')
    ax2.set_ylim((0,1.1*NU))
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay, color='tab:gray')    
    ax2.tick_params(axis='y', labelcolor='black')
    ax22.tick_params(axis='y', labelcolor='tab:gray')
    ax2.set_ylabel('Dissemination rate (Work/sec)', color='black')
    ax22.set_ylabel('Mean Latency (sec)', color='tab:gray')
    fig2.tight_layout()
    plt.savefig(dirstr+'/Throughput.png', bbox_inches='tight')
    
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.grid(linestyle='--')
    ax3.set_xlabel('Latency (sec)')
    plot_cdf(latencies, ax3)
    plt.savefig(dirstr+'/Latency.png', bbox_inches='tight')
    
    '''
    fig3a, ax3a = plt.subplots(figsize=(8,4))
    ax3a.grid(linestyle='--')
    ax3a.set_xlabel('Inbox Latency (sec)')
    plot_cdf(inboxLatencies, ax3a)
    plt.savefig(dirstr+'/InboxLatency.png', bbox_inches='tight')
    '''
    '''
    fig4, ax4 = plt.subplots(figsize=(8,4))
    ax4.grid(linestyle='--')
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel(r'$\lambda_i$')
    ax4.plot(np.arange(0, SIM_TIME, STEP), np.sum(avgLmds, axis=1), color='tab:blue')
    
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue')
        if MODE[NodeID]==2:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red')
        if MODE[NodeID]==3:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green')
    
    plt.savefig(dirstr+'/IssueRates.png', bbox_inches='tight')
    '''
    
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax5.grid(linestyle='--')
    ax5.set_xlabel('Time (sec)')
    ax5.set_ylabel('Reputation-scaled inbox length (neighbour)')
    N=100
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLen[:,NodeID], 'valid'), color='tab:blue', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]==2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLen[:,NodeID], 'valid'), color='tab:red', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]>2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLen[:,NodeID], 'valid'), color='tab:green', linewidth=5*REP[NodeID]/REP[0])
    ax5.set_xlim(0, SIM_TIME)
    
    plt.savefig(dirstr+'/AvgInboxLen.png', bbox_inches='tight')
    
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax5.grid(linestyle='--')
    ax5.set_xlabel('Time (sec)')
    ax5.set_ylabel('Reputation-scaled inbox length (neighbour of neighbour)')
    N=100
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLenMA[:,NodeID], 'valid'), color='tab:blue', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]==2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLenMA[:,NodeID], 'valid'), color='tab:red', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]>2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgInboxLenMA[:,NodeID], 'valid'), color='tab:green', linewidth=5*REP[NodeID]/REP[0])
    ax5.set_xlim(0, SIM_TIME)
    
    plt.savefig(dirstr+'/AvgInboxLenMA.png', bbox_inches='tight')
    """
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax5.grid(linestyle='--')
    ax5.set_xlabel('Time (sec)')
    ax5.set_ylabel('Deficits at Node 6')
    N=100
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgSolReq[:,NodeID], 'valid'), color='tab:blue', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]==2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgSolReq[:,NodeID], 'valid'), color='tab:red', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]>2:
            ax5.plot(np.arange((N-1)*STEP, SIM_TIME, STEP), np.convolve(np.ones(N)/N, avgSolReq[:,NodeID], 'valid'), color='tab:green', linewidth=5*REP[NodeID]/REP[0])
    ax5.set_xlim(0, SIM_TIME)
    
    plt.savefig(dirstr+'/AvgSolReq.png', bbox_inches='tight')
    """
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax5.grid(linestyle='--')
    ax5.set_xlabel('Time (sec)')
    ax5.set_ylabel('Deficits at Node 6')
    N=100
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax5.plot(np.arange(0, SIM_TIME, STEP), avgDefs[:,NodeID], color='tab:blue', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]==2:
            ax5.plot(np.arange(0, SIM_TIME, STEP), avgDefs[:,NodeID], color='tab:red', linewidth=5*REP[NodeID]/REP[0])
        if MODE[NodeID]>2:
            ax5.plot(np.arange(0, SIM_TIME, STEP), avgDefs[:,NodeID], color='tab:green', linewidth=5*REP[NodeID]/REP[0])
    ax5.set_xlim(0, SIM_TIME)
    
    plt.savefig(dirstr+'/avgDefs.png', bbox_inches='tight')
    '''
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
    
    plt.savefig(dirstr+'/InboxLenMA.png', bbox_inches='tight')
    '''
    fig6, ax6 = plt.subplots(figsize=(8,4))
    ax6.grid(linestyle='--')
    ax6.set_xlabel('Node ID')
    ax6.title.set_text('Reputation Distribution')
    ax6.set_ylabel('Reputation')
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==0:
            ax6.bar(NodeID, REP[NodeID], color='gray')
        if MODE[NodeID]==1:
            ax6.bar(NodeID, REP[NodeID], color='tab:blue')
        if MODE[NodeID]==2:
            ax6.bar(NodeID, REP[NodeID], color='tab:red')
        if MODE[NodeID]>2:
            ax6.bar(NodeID, REP[NodeID], color='tab:green')
    ModeLines = [Line2D([0],[0],color='tab:red', lw=4), Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='gray', lw=4), Line2D([0],[0],color='tab:green', lw=4)]
    ax6.legend(ModeLines, ['Best-effort', 'Content', 'Inactive', 'Malicious'], loc='upper right')
    plt.savefig(dirstr+'/RepDist.png', bbox_inches='tight')
    '''
    fig7, ax7 = plt.subplots(figsize=(8,4))
    plot_cdf(ISTimes, ax7)
    ax7.grid(linestyle='--')
    ax7.set_xlabel('Inter-service time (sec)')
    plt.savefig(dirstr+'/InterServiceTimes.png', bbox_inches='tight')
    '''
    '''
    fig8, ax8 = plt.subplots(figsize=(8,4))
    #plot_cdf_exp(IATimes, ax8)
    ax8.grid(linestyle='--')
    ax8.set_xlabel('Inter-arrival time (sec)')
    plt.savefig(dirstr+'/InterArrivalTimes.png', bbox_inches='tight')
    '''
    fig9, ax9 = plt.subplots(figsize=(8,4))
    ax9.grid(linestyle='--')
    ax9.plot(np.arange(0, SIM_TIME, STEP), avgOTA, color='black')
    ax9.set_ylabel('Max time in transit (sec)')
    ax9.set_xlabel('Time (sec)')
    plt.savefig(dirstr+'/MaxAge.png', bbox_inches='tight')

def plot_cdf(data, ax, xlim=0):
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
            if MODE[NodeID]>2:
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
        
    return maxval
    
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

class Transaction:
    """
    Object to simulate a transaction and its edges in the DAG
    """
    def __init__(self, IssueTime, Parents, Node, Work=0, Index=None, VisibleTime=None):
        self.IssueTime = IssueTime
        self.VisibleTime = VisibleTime
        self.Children = []
        self.Parents = Parents
        for p in Parents:
            p.Children.append(self)
        self.Index = Index
        self.InformedNodes = 0
        self.GlobalSolidTime = []
        self.Work = Work
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
        else: # genesis
            self.NodeID = []
            
class SolRequest:
    '''
    Object to request solidification of a transaction
    '''
    def __init__(self, Tran):
        self.Tran = Tran

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = []
        self.Ledger = [Genesis]
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox(self)
        self.NodeID = NodeID
        self.Alpha = ALPHA*REP[NodeID]/sum(REP)
        self.Lambda = NU*REP[NodeID]/sum(REP)
        self.BackOff = []
        self.LastBackOff = []
        self.LastScheduleTime = 0
        self.LastScheduleWork = 0
        self.LastIssueTime = 0
        self.LastIssueWork = 0
        self.IssuedTrans = []
        self.Undissem = 0
        self.UndissemWork = 0
        self.ServiceTimes = []
        self.ArrivalTimes = []
        self.ArrivalWorks = []
        self.InboxLatencies = []
        self.ValidTips = [True for NodeID in range(NUM_NODES)]
        self.TranCounter = 0
        self.Blacklist = []
        
    def issue_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2:
                if self.BackOff:
                    self.LastIssueTime += TAU#BETA*REP[self.NodeID]/self.Lambda
                while Time+STEP >= self.LastIssueTime + self.LastIssueWork/self.Lambda:
                    self.LastIssueTime += self.LastIssueWork/self.Lambda
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    if IOT[self.NodeID]:
                        Work = np.random.uniform(IOTLOW,IOTHIGH)
                    else:
                        Work = 1
                    self.LastIssueWork = Work
                    self.TranCounter += 1
                    self.IssuedTrans.append(Transaction(self.LastIssueTime, Parents, self, Work, Index=self.TranCounter))
            elif MODE[self.NodeID]==1:
                if IOT[self.NodeID]:
                    Work = np.random.uniform(IOTLOW,IOTHIGH)
                else:
                    Work = 1
                times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
                for t in times:
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.TranCounter += 1
                    self.IssuedTrans.append(Transaction(t, Parents, self, Work, Index=self.TranCounter))
            else:
                Work = 1
                times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
                for t in times:
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.TranCounter += 1
                    self.IssuedTrans.append(Transaction(t, Parents, self, Work, Index=self.TranCounter))
                
        # check PoW completion
        while self.IssuedTrans:
            Tran = self.IssuedTrans.pop(0)
            p = Packet(self, self, Tran, Tran.IssueTime, Tran.IssueTime)
            if MODE[self.NodeID]>2: # malicious don't consider own txs for scheduling
                self.add_to_ledger(self, Tran, Tran.IssueTime)
            else:
                self.add_to_inbox(p, Tran.IssueTime)
    
    def select_tips(self):
        """
        Implements uniform random tip selection
        """
        ValidTips = [tip for tip in self.TipsSet if self.ValidTips[tip.NodeID]]
        if len(ValidTips)>1:
            Selection = sample(ValidTips, 2)
        elif len(self.Ledger)<2:
            Selection = [self.Ledger[0]]
        else:
            Selection = self.Ledger[-2:-1]
        return Selection
    
    def schedule_txs(self, Time):
        """
        schedule txs from inbox at a fixed deterministic rate NU
        """
        # sort inboxes by arrival time
        self.Inbox.AllPackets.sort(key=lambda p: p.EndTime)
        self.Inbox.SolidPackets.sort(key=lambda p: p.EndTime)
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.Data.IssueTime)

        # process according to global rate Nu
        while self.Inbox.SolidPackets or self.Inbox.Scheduled:
            if self.Inbox.Scheduled:
                nextSchedTime = self.LastScheduleTime+(self.LastScheduleWork/NU)
            else:
                nextSchedTime = max(self.LastScheduleTime+(self.LastScheduleWork/NU), self.Inbox.SolidPackets[0].EndTime)
                
            if nextSchedTime<Time+STEP:             
                if SCHEDULING=='drr':
                    Packet = self.Inbox.drr_schedule(nextSchedTime)
                elif SCHEDULING=='drr_lds':
                    Packet = self.Inbox.drr_lds_schedule(nextSchedTime)
                elif SCHEDULING=='drrpp':
                    Packet = self.Inbox.drrpp_schedule(nextSchedTime)
                elif SCHEDULING=='fifo':
                    Packet = self.Inbox.fifo_schedule(nextSchedTime)
                if Packet is not None:
                    if Packet.Data not in self.Ledger:
                        self.add_to_ledger(Packet.TxNode, Packet.Data, nextSchedTime)
                    # update AIMD
                    #if Packet.Data.NodeID==self.NodeID:
                    self.Network.Nodes[Packet.Data.NodeID].InboxLatencies.append(nextSchedTime-Packet.EndTime)
                    self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*sum([p.Data.Work for p in self.Inbox.Packets[self.NodeID]])
                    self.set_rate(nextSchedTime)
                    self.LastScheduleTime = nextSchedTime
                    self.LastScheduleWork = Packet.Data.Work
                    self.ServiceTimes.append(nextSchedTime)
                else:
                    break
            else:
                break
    
    def add_to_ledger(self, TxNode, Tran, Time):
        """
        Adds the transaction to the local copy of the ledger and broadcast it
        """
        self.Ledger.append(Tran)
        if Tran.NodeID==self.NodeID:
            self.Undissem += 1
            self.UndissemWork += Tran.Work
            Tran.VisibleTime = Time
        # mark this TX as received by this node
        Tran.InformedNodes += 1
        if Tran.InformedNodes==NUM_NODES:
            self.Network.Throughput[Tran.NodeID] += 1
            self.Network.WorkThroughput[Tran.NodeID] += Tran.Work
            self.Network.TranDelays.append(Time-Tran.IssueTime)
            self.Network.VisTranDelays.append(Time-Tran.VisibleTime)
            self.Network.DissemTimes.append(Time)
            Tran.GlobalSolidTime = Time
            self.Network.Nodes[Tran.NodeID].Undissem -= 1
            self.Network.Nodes[Tran.NodeID].UndissemWork -= Tran.Work
        
        if Tran.Children:
            if not [c for c in Tran.Children if c in self.Ledger]:
                self.TipsSet.append(Tran)
            for c in Tran.Children:
                if c in self.Inbox.Trans:
                    if self.is_solid(c):
                        packet = [p for p in self.Inbox.AllPackets if p.Data==c]
                        self.Inbox.SolidPackets.append(packet[0])
        else:
            self.TipsSet.append(Tran)

        for Parent in Tran.Parents:
            if Parent in self.TipsSet:
                self.TipsSet.remove(Parent)
            else:
                continue
        self.forward(TxNode, Tran, Time)# broadcast the packet
        
    def forward(self, TxNode, Tran, Time):
        '''
        Forward this transaction to all but the node it was received from
        '''
        if MODE[self.NodeID]==4:
            neighb = np.random.randint(len(self.Neighbours))
            self.Network.send_data(self, self.Neighbours[neighb], Tran, Time)
        else:
            self.Network.broadcast_data(self, TxNode, Tran, Time)
            
    def check_congestion(self, Time):
        """
        Check for rate setting
        """
        if self.Inbox.Avg>MIN_TH*REP[self.NodeID]:
            if self.Inbox.Avg>MAX_TH*REP[self.NodeID]:
                self.BackOff = True
            elif np.random.rand()<P_B*(self.Inbox.Avg-MIN_TH*REP[self.NodeID])/((MAX_TH-MIN_TH)*REP[self.NodeID]):
                self.BackOff = True
            
    def set_rate(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2 and Time>=START_TIMES[self.NodeID]: # AIMD starts after settling time
                # if wait time has not passed---reset.
                if self.LastBackOff:
                    if Time < self.LastBackOff + TAU:#BETA*REP[self.NodeID]/self.Lambda:
                        self.BackOff = False
                        return
                # multiplicative decrease or else additive increase
                if self.BackOff:
                    self.Lambda = self.Lambda*BETA
                    self.BackOff = False
                    self.LastBackOff = Time
                else:
                    self.Lambda += self.Alpha
            elif MODE[self.NodeID]>2 and Time>=START_TIMES[self.NodeID]: # malicious starts after start time
                self.Lambda = NUM_NEIGHBOURS*NU*REP[self.NodeID]/sum(REP)
            else: # any active node, malicious or honest
                self.Lambda = NU*REP[self.NodeID]/sum(REP)
        else:
            self.Lambda = 0
            
    def add_to_inbox(self, Packet, Time):
        """
        Add to inbox if not already received and/or processed
        """
        Tran = Packet.Data
        if Tran not in self.Inbox.Trans:
            if Tran not in self.Ledger:
                NodeID = Tran.NodeID
                ScaledWork = self.Inbox.Work[NodeID]/REP[NodeID]
                if Tran not in self.Inbox.RequestedTrans:
                    if NodeID not in self.Blacklist:
                        if NodeID==self.NodeID:
                            self.Inbox.add_packet(Packet)
                            self.check_congestion(Time)
                        else:
                            if ScaledWork>DROP_TH and BLACKLIST:
                                self.Blacklist.append(NodeID)
                                self.Inbox.drop_packet(Packet)
                            else:
                                self.ValidTips[NodeID] = True
                                self.Inbox.add_packet(Packet)
                    else:
                        self.Inbox.drop_packet(Packet)
                else:
                    self.Inbox.add_packet(Packet)
                
                self.ArrivalWorks.append(Tran.Work)
                self.ArrivalTimes.append(Time)
                            
    def is_solid(self, Tran):
        isSolid = True
        for p in Tran.Parents:
            if p not in self.Ledger:
                isSolid = False
        return isSolid
    
class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self, Node):
        self.Node = Node
        self.AllPackets = [] # Inbox_m
        self.SolidPackets = []
        self.Packets = [[] for NodeID in range(NUM_NODES)] # Inbox_m(i)
        self.Work = np.zeros(NUM_NODES)
        self.Trans = []
        self.RRNodeID = np.random.randint(NUM_NODES) # start at a random node
        self.Deficit = np.zeros(NUM_NODES)
        self.Scheduled = []
        self.Avg = 0
        self.New = []
        self.Old = []
        self.Empty = [NodeID for NodeID in range(NUM_NODES)]
        self.RequestedTrans = []
        self.DroppedTrans = []
        self.DropTimes = []
    
    def add_packet(self, Packet):
        Tran = Packet.Data
        NodeID = Tran.NodeID
        if Tran in self.RequestedTrans:
            if self.Packets[NodeID]:
                Packet.EndTime = self.Packets[NodeID][0].EndTime # move packet to the front of the queue
            self.RequestedTrans = [tran for tran in self.RequestedTrans if tran != Tran]
            self.Packets[NodeID].insert(0,Packet)
        else:
            self.Packets[NodeID].append(Packet)
        self.AllPackets.append(Packet)
        # check if solid
        if self.Node.is_solid(Tran):
            self.SolidPackets.append(Packet)
        if NodeID in self.Empty:
            self.Empty.remove(NodeID)
            self.New.append(NodeID)
            self.Deficit[NodeID] += QUANTUM[NodeID]
        self.Trans.append(Tran)
        self.Work[NodeID] += Packet.Data.Work
    
    def drop_packet(self, Packet):
        self.DroppedTrans.append(Packet.Data)
        self.DropTimes.append(Packet.EndTime)
    
    def remove_packet(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.Trans:
            if Packet in self.AllPackets:
                Tran = Packet.Data # Take transaction from the packet
                self.AllPackets.remove(Packet)
                if Packet in self.SolidPackets:
                    self.SolidPackets.remove(Packet)
                self.Packets[Packet.Data.NodeID].remove(Packet)
                self.Trans.remove(Tran)  
                self.Work[Packet.Data.NodeID] -= Packet.Data.Work
        
    def drr_schedule(self, Time): # Not up to date with solidification rules
        if self.Scheduled:
            return self.Scheduled.pop(0)
        if self.AllPackets:
            while not self.Scheduled:
                if self.Packets[self.RRNodeID]:
                    self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
                    i=0
                    while self.Packets[self.RRNodeID] and i<len(self.Packets[self.RRNodeID]):
                        Tran = self.Packets[self.RRNodeID][i].Data
                        Work = Tran.Work
                        if not self.is_solid(Tran):
                            continue
                        if self.Deficit[self.RRNodeID]>=Work:
                            Packet = self.Packets[self.RRNodeID][i]
                            self.Deficit[self.RRNodeID] -= Work
                            # remove the transaction from all inboxes
                            self.remove_packet(Packet)
                            self.Scheduled.append(Packet)
                        else:
                            break
                    self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
                else: # move to next node and assign deficit
                    self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
            return self.Scheduled.pop(0)
    
    def drr_lds_schedule(self, Time):
        if self.Scheduled:
            return self.Scheduled.pop(0)
        
        if SCHEDULE_ON_SOLID:
            Packets = [self.SolidPackets]
        else:
            Packets = [self.AllPackets]
        while Packets[0] and not self.Scheduled:
            if self.Deficit[self.RRNodeID]<MAX_WORK:
                self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
            i = 0
            while self.Packets[self.RRNodeID] and i<len(self.Packets[self.RRNodeID]):
                Packet = self.Packets[self.RRNodeID][i]
                if Packet not in Packets[0]:
                    i += 1
                    if SOLID_REQUESTS:
                        for p in Packet.Data.Parents:
                            if p not in self.RequestedTrans and p not in self.AllPackets:
                                # send a solidification request for this tran's parents
                                self.Node.Network.send_data(self.Node, Packet.TxNode, SolRequest(p), Time)
                                self.RequestedTrans.append(p)
                    continue
                Work = Packet.Data.Work
                if self.Deficit[self.RRNodeID]>=Work and Packet.EndTime<=Time:
                    self.Deficit[self.RRNodeID] -= Work
                    # remove the transaction from all inboxes
                    self.remove_packet(Packet)
                    self.Scheduled.append(Packet)
                else:
                    i += 1
                    continue
            self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
        if self.Scheduled:
            return self.Scheduled.pop(0)
        
    def drrpp_schedule(self, Time): # Not up to date
        while self.New:
            NodeID = self.New[0]
            if not self.Packets[NodeID] or self.Deficit[NodeID]<0:
                self.New.remove(NodeID)
                self.Old.append(NodeID)
                continue
            Work = self.Packets[NodeID][0].Data.Work
            Packet = self.Packets[NodeID][0]
            self.Deficit[NodeID] -= Work
            # remove the transaction from all inboxes
            self.remove_packet(Packet)
            return Packet
        while self.Old:
            NodeID = self.Old[0]
            if self.Packets[NodeID]:
                Work = self.Packets[NodeID][0].Data.Work
                if self.Deficit[NodeID]>=Work:
                    Packet = self.Packets[NodeID][0]
                    self.Deficit[NodeID] -= Work
                    # remove the transaction from all inboxes
                    self.remove_packet(Packet)
                    return Packet
                else:
                    # move this node to end of list
                    self.Old.remove(NodeID)
                    self.Old.append(NodeID)
                    self.Deficit[self.Old[0]] += QUANTUM[self.Old[0]]
            else:
                self.Old.remove(NodeID)
                if self.Deficit[NodeID]<0:
                    self.Old.append(NodeID)
                else:
                    self.Empty.append(NodeID)
                if self.Old:
                    self.Deficit[self.Old[0]] += QUANTUM[self.Old[0]]
                    
    def fifo_schedule(self, Time): # Not up to date with solidification rules
        if self.AllPackets:
            Packet = self.AllPackets[0]
            # remove the transaction from all inboxes
            self.remove_packet(Packet)
            return Packet

class Packet:
    """
    Object for sending data including TXs and back off notifications over
    comm channels
    """    
    def __init__(self, TxNode, RxNode, Data, StartTime, EndTime = []):
        # can be a TX or a back off notification
        self.TxNode = TxNode
        self.RxNode = RxNode
        self.Data = Data
        self.StartTime = StartTime
        self.EndTime = EndTime
    
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
        self.PacketDelays = []
    
    def send_packet(self, TxNode, RxNode, Data, Time):
        """
        Add new packet to the comm channel with time of arrival
        """
        self.Packets.append(Packet(TxNode, RxNode, Data, Time))
        self.PacketDelays.append(np.random.exponential(scale=self.Delay))
    
    def transmit_packets(self, Time):
        """
        Move packets through the comm channel, simulating delay
        """
        if self.Packets:
            for Packet in self.Packets:
                i = self.Packets.index(Packet)
                if(self.Packets[i].StartTime+self.PacketDelays[i]<=Time):
                    self.deliver_packet(self.Packets[i], self.Packets[i].StartTime+self.PacketDelays[i])
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
        elif isinstance(Packet.Data, SolRequest):
            # else if this is a solidification request, retrieve the transaction and send it back
            Tran = Packet.Data.Tran
            self.RxNode.Network.send_data(self.RxNode, self.TxNode, Tran, Time)
        else: 
            # else this is a back off notification
            self.RxNode.process_cong_notif(Packet, Time)
        PacketIndex = self.Packets.index(Packet)
        self.Packets.remove(Packet)
        del self.PacketDelays[PacketIndex]
        
class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix):
        self.A = AdjMatrix
        self.Nodes = []
        self.CommChannels = []
        self.Throughput = [0 for NodeID in range(NUM_NODES)]
        self.WorkThroughput = [0 for NodeID in range(NUM_NODES)]
        self.TranDelays = []
        self.VisTranDelays = []
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
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],np.random.exponential(0.1)))
            self.CommChannels.append(RowList)
    
    def remove_node(self, Node):
        Node.Neighbours = []
        self.CommChannels[Node.NodeID] = []
    
    def add_node(self, Node):
        EligNeighbs = [n for n in self.Nodes if (Node.NodeID not in n.Blacklist) and (n is not Node)]
        if EligNeighbs:
            print(len(EligNeighbs))
            NewNeighbours = np.random.choice(EligNeighbs, size=min(len(EligNeighbs),NUM_NEIGHBOURS), replace=False)
            for Neighbour in NewNeighbours:
                Node.Neighbours.append(Neighbour)
                Neighbour.Neighbours.append(Node)
                self.CommChannels[Node.NodeID].append(CommChannel(Node, Neighbour, np.random.exponential(0.1)))
                self.CommChannels[Neighbour.NodeID].append(CommChannel(Neighbour, Node, np.random.exponential(0.1)))
            
    
    def send_data(self, TxNode, RxNode, Data, Time):
        """
        Send this data (TX or back off) to a specified neighbour
        """
        if RxNode in TxNode.Neighbours:
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
        Each node generate new transactions
        """
        for Node in self.Nodes:
            Node.issue_txs(Time)
            if MODE[Node.NodeID]>2:
                if len([Neighbour for Neighbour in Node.Neighbours if Node.NodeID in Neighbour.Blacklist])==NUM_NEIGHBOURS:
                    self.remove_node(Node)
                    self.add_node(Node)
                
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time+STEP)
        """
        Each node schedule transactions in inbox
        """
        for Node in self.Nodes:
            Node.schedule_txs(Time)
    
    def tran_latency(self, latencies, latTimes):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.IssueTime>20:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.IssueTime)
                latTimes[Tran.NodeID].append(Tran.GlobalSolidTime)
        return latencies, latTimes
                
if __name__ == "__main__":
        main()
        
    
    