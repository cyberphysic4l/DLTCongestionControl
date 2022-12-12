# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import os
import shutil
from time import gmtime, strftime
from core.global_params import *
from core.network import Network
from utils import all_node_plot, per_node_barplot, per_node_plot, per_node_plot_mean, plot_cdf, per_node_rate_plot, scaled_rate_plot
    
def main():
    '''
    Create directory for storing results with these parameters
    '''
    per_node_result_keys = ['AllReadyPackets',
                            'Dropped Messages',
                            'Number of Tips',
                            'Number of Honest Tips',
                            'Inbox Lengths',
                            'Inbox Lengths (moving average)',
                            'Deficits',
                            'Number of Disseminated Messages',
                            'Number of Undisseminated Messages',
                            'Number of Confirmed Messages',
                            'Number of Unconfirmed Messages',
                            'Number of Scheduled Messages']
    dirstr1 = os.path.dirname(os.path.realpath(__file__)) 
    dirstr = dirstr1 + '/results/'+ strftime("%Y-%m-%d_%H%M%S", gmtime())
    os.makedirs(dirstr, exist_ok=True)
    os.makedirs(dirstr+'/raw', exist_ok=True)
    os.makedirs(dirstr+'/plots', exist_ok=True)
    shutil.copy("core/global_params.py", dirstr+"/global_params.txt")
    per_node_result_keys = simulate(per_node_result_keys, dirstr)
    plot_results(dirstr, per_node_result_keys)
    
def simulate(per_node_result_keys, dirstr):
    """
    Setup simulation inputs and instantiate output arrays
    """
    # seed rng
    np.random.seed(1)
    TimeSteps = int(SIM_TIME/STEP)
    
    """
    Monte Carlo Sims
    """
    PacketsInTransit = [np.zeros(TimeSteps) for mc in range(MONTE_CARLOS)]
    Lmds = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    OldestTxAges = np.zeros((TimeSteps, NUM_NODES))
    OldestTxAge = []
    OldestUnconfAge = [np.zeros(TimeSteps) for mc in range(MONTE_CARLOS)]
    Droppees = {}
    per_node_results = {}
    for k in per_node_result_keys:
        per_node_results[k] = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    MeanDelay = [np.zeros(int(TimeSteps/100)) for mc in range(MONTE_CARLOS)]
    ConfDelay = [np.zeros(int(TimeSteps/100)) for mc in range(MONTE_CARLOS)]
    Unsolid = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    EligibleDelays = [np.zeros((SIM_TIME, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    latencies = [[] for NodeID in range(NUM_NODES)]
    confLatencies = [[] for NodeID in range(NUM_NODES)]
    latTimes = [[] for NodeID in range(NUM_NODES)]
    confLatTimes = [[] for NodeID in range(NUM_NODES)]
    ServTimes = [[] for NodeID in range(NUM_NODES)]
    ArrTimes = [[] for NodeID in range(NUM_NODES)]
    interArrTimes = [[] for NodeID in range(NUM_NODES)]
    h_latency = [[] for NodeID in range(NUM_NODES)]
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
        MalNeighb = Net.Nodes[2].Neighbours[0]
        for i in range(len(MalNeighb.Neighbours)):
            if MalNeighb.Neighbours[i] not in Net.Nodes[2].Neighbours and MalNeighb.Neighbours[i].NodeID!=2:
                NonMalNeighb = MalNeighb.Neighbours[i]
                break
        # output arrays
        for i in range(TimeSteps):
            if 100*i/TimeSteps%10==0:
                print("Simulation: "+str(mc+1) +"\t " + str(int(100*i/TimeSteps))+"% Complete")
            # discrete time step size specified by global variable STEP
            T = STEP*i
            """
            The next line is the function which ultimately calls all others
            and runs the simulation for a time step
            """
            Net.simulate(T)
            ages = [T-fc for _,fc in Net.FirstConfTimes.items()]
            if ages:
                OldestUnconfAge[mc][i] = max(ages)
            else:
                OldestUnconfAge[mc][i] = 0
            # save summary results in output arrays
            PacketsInTransit[mc][i] = sum([sum([len(cc.Packets) for cc in ccs]) for ccs in Net.CommChannels])
            for NodeID, Node in enumerate(Net.Nodes):
                Lmds[mc][i, NodeID] = min(Node.Lambda, Node.LambdaD)
                
                per_node_results['Inbox Lengths'][mc][i,NodeID] = len(Node.Inbox.AllPackets)
                per_node_results['AllReadyPackets'][mc][i,NodeID] = len(Node.Inbox.AllReadyPackets)
                per_node_results['Dropped Messages'][mc][i,NodeID] = sum([len(Node.DroppedPackets[i]) for i in range(NUM_NODES)])
                if sum([len(n.DroppedPackets[NodeID]) for n in Net.Nodes]):
                    Droppees[NodeID] = sum([len(n.DroppedPackets[NodeID]) for n in Net.Nodes])
                per_node_results['Number of Tips'][mc][i,NodeID] = len(Node.TipsSet)
                per_node_results['Number of Honest Tips'][mc][i,NodeID] = sum([len(Node.NodeTipsSet[i]) for i in range(NUM_NODES) if MODE[i]<3])
                per_node_results['Inbox Lengths (moving average)'][mc][i,NodeID] = Node.Inbox.Avg
                per_node_results['Deficits'][mc][i, NodeID] = Net.Nodes[0].Inbox.Deficit[NodeID]
                per_node_results['Number of Disseminated Messages'][mc][i, NodeID] = Net.Disseminated[NodeID]
                per_node_results['Number of Scheduled Messages'][mc][i, NodeID] = Net.Scheduled[NodeID]
                per_node_results['Number of Undisseminated Messages'][mc][i,NodeID] = Node.Undissem
                per_node_results['Number of Confirmed Messages'][mc][i,NodeID] = len(Node.ConfMsgs)
                per_node_results['Number of Unconfirmed Messages'][mc][i,NodeID] = len(Node.UnconfMsgs)
        print("Simulation: "+str(mc+1) +"\t 100% Complete")
        
        OldestTxAge.append(np.mean(OldestTxAges, axis=1))
        for i in range(int(TimeSteps/100)):
            s = STEP*100
            delays = [Net.MsgDelays[j] for j in Net.MsgDelays if s*int(Net.DissemTimes[j]/s)==i*s and MODE[Net.MsgIssuer[j]]<3]
            if delays:
                MeanDelay[mc][i] = sum(delays)/len(delays)
            confDelays = [Net.ConfTimes[j]-Net.Nodes[0].Ledger[j].IssueTime for j in Net.ConfTimes if s*int(Net.ConfTimes[j]/s)==i*s]
            if confDelays:
                ConfDelay[mc][i] = sum(confDelays)/len(confDelays)
        for NodeID in range(NUM_NODES):
            for i in range(SIM_TIME):
                delays = []
                for _,msg in Net.Nodes[NodeID].Ledger.items():
                    if msg.EligibleTime is not None and msg.NodeID:
                        if int(msg.EligibleTime)==i and MODE[msg.NodeID]<3: # don't count malicious msg delays.
                            delays.append(msg.EligibleTime-msg.IssueTime)
                if delays:
                    EligibleDelays[mc][i, NodeID] = sum(delays)/len(delays)
                
            ServTimes[NodeID] = sorted(Net.Nodes[NodeID].ServiceTimes)
            ArrTimes[NodeID] = sorted(Net.Nodes[NodeID].ArrivalTimes)
            ArrWorks = [x for _,x in sorted(zip(Net.Nodes[NodeID].ArrivalTimes,Net.Nodes[NodeID].ArrivalWorks))]
            interArrTimes[NodeID].extend(np.diff(ArrTimes[NodeID])/ArrWorks[1:])
            h_latency[NodeID].extend(Net.Nodes[NodeID].TipsSetDelay)
                
        latencies, latTimes = Net.msg_latency(latencies, latTimes)
        confLatencies, confLatTimes = Net.msg_conf_latency(confLatencies, confLatTimes)

        del Net
    """
    Get results
    """
    print(Droppees)
    avg_per_node_results = {}
    for k in per_node_results:
        avg_per_node_results[k] = sum(per_node_results[k])/len(per_node_results[k])

    avgPIT = sum(PacketsInTransit)/len(PacketsInTransit)
    avgLmds = sum(Lmds)/len(Lmds)
    avgMeanDelay = sum(MeanDelay)/len(MeanDelay)
    avgConfDelay = sum(ConfDelay)/len(ConfDelay)
    avgUnsolid = sum(Unsolid)/len(Unsolid)
    avgEligibleDelays = sum(EligibleDelays)/len(EligibleDelays)
    avgOTA = sum(OldestTxAge)/len(OldestTxAge)
    avgOUA = sum(OldestUnconfAge)/len(OldestUnconfAge)
    """
    Create a directory for these results and save them
    """
    np.savetxt(dirstr+'/raw/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/raw/avgPIT.csv', avgPIT, delimiter=',')
    for k in per_node_results:
        np.savetxt(dirstr+'/raw/' + k + '.csv', avg_per_node_results[k], delimiter=',')
    np.savetxt(dirstr+'/raw/avgMeanDelay.csv', avgMeanDelay, delimiter=',')
    np.savetxt(dirstr+'/raw/avgConfDelay.csv', avgConfDelay, delimiter=',')
    np.savetxt(dirstr+'/raw/avgOldestTxAge.csv', avgOTA, delimiter=',')
    np.savetxt(dirstr+'/raw/avgOldestUnconfAge.csv', avgOUA, delimiter=',')
    np.savetxt(dirstr+'/raw/avgUnsolid.csv', avgUnsolid, delimiter=',')
    np.savetxt(dirstr+'/raw/avgEligibleDelays.csv', avgEligibleDelays, delimiter=',')
    for NodeID in range(NUM_NODES):
        np.savetxt(dirstr+'/raw/confLatencies'+str(NodeID)+'.csv',
                   np.asarray(confLatencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/latencies'+str(NodeID)+'.csv',
                   np.asarray(latencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/h_latency'+str(NodeID)+'.csv',
                   np.asarray(h_latency[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/ServTimes'+str(NodeID)+'.csv',
                   np.asarray(ServTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/raw/ArrTimes'+str(NodeID)+'.csv',
                   np.asarray(ArrTimes[NodeID]), delimiter=',')
    nx.write_adjlist(G, dirstr+'/raw/result_adjlist.txt', delimiter=' ')
    return per_node_results.keys()


    
def plot_results(dirstr, per_node_result_keys):
    """
    Initialise plots
    """
    plt.close('all')
    
    """
    Load results from the data directory
    """
    per_node_results = {}
    for k in per_node_result_keys:
        per_node_results[k] = np.loadtxt(dirstr+'/raw/' + k + '.csv', delimiter=',')
    avgUnsolid = np.loadtxt(dirstr+'/raw/avgUnsolid.csv', delimiter=',')
    avgEligibleDelays = np.loadtxt(dirstr+'/raw/avgEligibleDelays.csv', delimiter=',')
    avgMeanDelay = np.loadtxt(dirstr+'/raw/avgMeanDelay.csv', delimiter=',')
    avgConfDelay = np.loadtxt(dirstr+'/raw/avgConfDelay.csv', delimiter=',')
    avgOTA = np.loadtxt(dirstr+'/raw/avgOldestTxAge.csv', delimiter=',')
    avgOUA = np.loadtxt(dirstr+'/raw/avgOldestUnconfAge.csv', delimiter=',')
    latencies = []
    h_latency = []
    confLatencies = []
    ServTimes = []
    ArrTimes = []
    
    for NodeID in range(NUM_NODES):
        if os.stat(dirstr+'/raw/latencies'+str(NodeID)+'.csv').st_size != 0:
            lat = [np.loadtxt(dirstr+'/raw/latencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            lat = [0]
        latencies.append(lat)
        if os.stat(dirstr+'/raw/confLatencies'+str(NodeID)+'.csv').st_size != 0:
            confLat = [np.loadtxt(dirstr+'/raw/confLatencies'+str(NodeID)+'.csv', delimiter=',')]
        else:
            confLat = [0]
        confLatencies.append(confLat)
        #if os.stat(dirstr+'/raw/h_latency'+str(NodeID)+'.csv').st_size != 0:
        #    lat = [np.loadtxt(dirstr+'/raw/h_latency'+str(NodeID)+'.csv', delimiter=',')]
        #else:
        #    lat = [0]
        #h_latency.append(lat)
        #ServTimes.append([np.loadtxt(dirstr+'/raw/ServTimes'+str(NodeID)+'.csv', delimiter=',')])
        #ArrTimes.append([np.loadtxt(dirstr+'/raw/ArrTimes'+str(NodeID)+'.csv', delimiter=',')])
    """
    Plot results
    """
    avg_window = 1000
    data = per_node_results['Number of Scheduled Messages']
    avgTP = np.concatenate((np.zeros((avg_window, NUM_NODES)),(data[avg_window:,:]-data[:-avg_window,:])))/(avg_window*STEP)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Time (sec)')
    HonestTP = sum(avgTP[avg_window:,NodeID] for NodeID in range(NUM_NODES))# if MODE[NodeID]<3)
    MaxHonestTP = NU#*sum([rep for i,rep in enumerate(REP) if MODE[i]<3])/sum(REP)
    ax2.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), 100*HonestTP/MaxHonestTP, color = 'black')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, STEP*100), avgMeanDelay, color='tab:gray')    
    ax2.tick_params(axis='y', labelcolor='black')
    ax22.tick_params(axis='y', labelcolor='tab:gray')
    ax2.set_ylabel(r'$DR/\nu \quad (\%)$', color='black')
    ax2.set_ylim([0,110])
    ax22.set_ylabel('Dissemination Latency (sec)', color='tab:gray')
    ModeLines = [Line2D([0],[0],color='black',linewidth=2), Line2D([0],[0],color='tab:gray', linewidth=2)]
    ax2.legend(ModeLines, ['Dissemination Rate', 'Dissemination Latency'], loc='lower right', ncol=1)
    #ax22.set_ylim([0,2])
    fig2.tight_layout()
    plt.savefig(dirstr+'/plots/Throughput.png', bbox_inches='tight')

    avg_window = 1000
    data = per_node_results['Number of Confirmed Messages']
    avgTP = np.concatenate((np.zeros((avg_window, NUM_NODES)),(data[avg_window:,:]-data[:-avg_window,:])))/(avg_window*STEP)
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Time (sec)')
    HonestTP = sum(avgTP[avg_window:,NodeID] for NodeID in range(NUM_NODES))# if MODE[NodeID]<3)
    MaxHonestTP = NU#*sum([rep for i,rep in enumerate(REP) if MODE[i]<3])/sum(REP)
    pc = HonestTP
    ax2.plot(np.arange(avg_window*STEP, SIM_TIME, STEP), pc, color = 'black')
    ax2.axhline(NU, xmax=SIM_TIME, color='tab:red', linestyle='--')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, STEP*100), avgConfDelay, color='tab:gray')
    ax2.tick_params(axis='y', labelcolor='black')
    ax22.tick_params(axis='y', labelcolor='tab:gray')
    ax2.set_ylabel('Confirmation Rate (blocks/sec)', color='black')
    ax2.set_ylim([0,1.05*max(pc)])
    ax22.set_ylabel('Confirmation Latency (sec)', color='tab:gray')
    ax22.set_ylim([0,2*max(avgConfDelay)])
    ModeLines = [Line2D([0],[0],color='black'), Line2D([0],[0],color='tab:red', linestyle='--'), Line2D([0],[0],color='tab:gray')]
    ax2.legend(ModeLines, ['Confirmation Rate', 'Scheduling Rate', 'Confirmation Latency'], loc='lower right', ncol=1)
    fig2.tight_layout()
    plt.savefig(dirstr+'/plots/ConfThroughput.png', bbox_inches='tight')
    

    plot_cdf(latencies, 'Latency (sec)', dirstr+'/plots/Latency.png')
    plot_cdf(confLatencies, 'Confrimation Latency (sec)', dirstr+'/plots/ConfLatency.png')
    
    #plot_cdf(h_latency, r'$H$ (sec)', dirstr+'/plots/H_latency_cdf.png')
    #plot_cdf_exp(h_latency, r'$H$ (sec)', dirstr+'/plots/H_latency_cdf_exp.png')

    #per_node_plot(avgLmds, 'Time (sec)', r'$\lambda_i$', '', dirstr, avg_window=1)
    
    for k in per_node_results:
        per_node_plot(per_node_results[k], 'Time (sec)', k, '', dirstr, avg_window=100)
    k = 'Number of Tips'
    per_node_plot_mean(per_node_results[k], 'Time (sec)', k, '', dirstr, avg_window=100)
    scaled_rate_plot(per_node_results['Number of Disseminated Messages'], 'Time (sec)', 'Dissemination Rate', 'Scaled Dissemination Rate', 'dissem', dirstr)
    scaled_rate_plot(per_node_results['Number of Confirmed Messages'], 'Time (sec)', r'$CR_i$ (blocks/sec)', r'$CR_i/\lambda_i$', 'conf', dirstr)
    scaled_rate_plot(per_node_results['Number of Scheduled Messages'], 'Time (sec)', 'Scheduling Rate', 'Scaled Scheduling Rate', 'sched', dirstr)

    per_node_rate_plot(per_node_results['Number of Disseminated Messages'], 'Time (sec)', 'Dissemination Rate', '', dirstr)
    per_node_rate_plot(per_node_results['Number of Confirmed Messages'], 'Time (sec)', 'Confirmation Rate', '', dirstr)
    per_node_rate_plot(per_node_results['Number of Scheduled Messages'], 'Time (sec)', 'Scheduling Rate', '', dirstr)

    per_node_plot(avgEligibleDelays, 'Time (sec)', 'Age of Messages Becoming Eligible', '', dirstr, avg_window=20, step=1)
    per_node_plot(avgUnsolid, 'Time (sec)', 'Unsolid', '', dirstr)
    
    per_node_barplot(REP, 'Node ID', 'Reputation', 'Reputation Distribution', dirstr+'/plots/RepDist.png')
    #per_node_barplot(QUANTUM, 'Node ID', 'Quantum', 'Quantum Distribution', dirstr+'/plots/QDist.png')

    all_node_plot(per_node_results['Number of Unconfirmed Messages'].sum(axis=1), 'Time (sec)', 'Number of Unconfirmed Messages', '', dirstr+'/plots/AllUnconfirmed.png')
    all_node_plot(avgOTA, 'Time (sec)', 'Max time in transit (sec)', '', dirstr+'/plots/MaxAge.png')
    all_node_plot(avgOUA, 'Time (sec)', 'Max time partially confirmed (sec)', '', dirstr+'/plots/MaxUnconfAge.png', avg_window=1000)

    """
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
    """

if __name__ == "__main__":
    main()