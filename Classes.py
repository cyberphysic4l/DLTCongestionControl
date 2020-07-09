# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from random import sample
from pathlib import Path
import winsound

np.random.seed(0)
# Simulation Parameters
MONTE_CARLOS = 1
SIM_TIME = 180
STEP = 0.01
# Network Parameters
NU = 50
NUM_NODES = 64
NUM_NEIGHBOURS = 4
START_TIMES = 10*np.ones(NUM_NODES)
REPDIST = 'zipf'
if REPDIST=='zipf':
    # IOTA data rep distribution - Zipf s=0.9
    REP = [(NUM_NODES+1)/((NodeID+1)**0.9) for NodeID in range(NUM_NODES)]
elif REPDIST=='uniform':
    # Permissioned System rep system?
    REP = np.ones(NUM_NODES, dtype=int)
# Modes: 0 = inactive, 1 = content, 2 = best-effort, 3 = malicious
if NUM_NODES%3==0:
    MODE = np.tile([2,1,0], int(NUM_NODES/3))
if NUM_NODES%3==1:
    MODE = np.concatenate((np.tile([2,1,0], int(NUM_NODES/3)), np.array([2])))
if NUM_NODES%3==2:
    MODE = np.concatenate((np.tile([2,1,0], int(NUM_NODES/3)), np.array([2,1])))
AVG_WORK = np.random.randint(1, 2, size=NUM_NODES)
MAX_WORK = 2.5

# Congestion Control Parameters
ALPHA = 0.1
BETA = 0.5
MAX_TH = 3
MIN_TH = 3
MAX_BURST = 1
QUANTUM = [rep/(sum(REP)) for rep in REP]
W_Q = 1
P_B = 0.5

SCHEDULING = 'drr_lds'
    
def main():
    '''
    Create directory for storing results with these parameters
    '''
    dirstr = 'data/sched='+SCHEDULING+'/rep='+REPDIST+'/nu='+str(NU)+'/alpha='+str(ALPHA)+'/beta='+str(BETA)+'/RED='+str(MIN_TH)+'_'+str(MAX_TH)+'/p_b='+str(P_B)+'/numnodes='+str(NUM_NODES)+'_' +str(NUM_NEIGHBOURS)+'/simtime='+str(SIM_TIME)+'/nmc='+str(MONTE_CARLOS)
    if not Path(dirstr).exists():
        print("Simulating")
        simulate(dirstr)
    else:
        print("Simulation already done for these parameters")
        simulate(dirstr)
    plot_results(dirstr)
    winsound.Beep(2500, 1000) # beep to say sim is finished
    
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
    Lmds = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    InboxLens = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Deficits = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Throughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    WorkThroughput = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    Undissem = [np.zeros((TimeSteps, NUM_NODES)) for mc in range(MONTE_CARLOS)]
    MeanDelay = [np.zeros(SIM_TIME) for mc in range(MONTE_CARLOS)]
    TP = []
    WTP = []
    latencies = [[] for NodeID in range(NUM_NODES)]
    latTimes = [[] for NodeID in range(NUM_NODES)]
    interServTimes = [[] for NodeID in range(NUM_NODES)]
    interArrTimes = [[] for NodeID in range(NUM_NODES)]
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
                InboxLens[mc][i, NodeID] = len(Net.Nodes[NodeID].Inbox.Packets[NodeID])
                Deficits[mc][i, NodeID] = Net.Nodes[0].Inbox.Deficit[NodeID]
                Throughput[mc][i, NodeID] = Net.Throughput[NodeID]
                WorkThroughput[mc][i,NodeID] = Net.WorkThroughput[NodeID]
                Undissem[mc][i,NodeID] = Net.Nodes[NodeID].Undissem
        
        for NodeID in range(NUM_NODES):
            for i in range(SIM_TIME):
                delays = [Net.TranDelays[j] for j in range(len(Net.TranDelays)) if int(Net.DissemTimes[j])==i]
                if delays:
                    MeanDelay[mc][i] = sum(delays)/len(delays)
            interServTimes[NodeID].append(np.diff(Net.Nodes[NodeID].ServiceTimes[NodeID]))
            interArrTimes[NodeID].append(np.diff(sorted(Net.Nodes[NodeID].ArrivalTimes)))
                
        latencies, latTimes = Net.tran_latency(latencies, latTimes)
        """
        for NodeID in range(NUM_NODES):
            if latencies[NodeID]:
                for i in range(SIM_TIME):
                    if latTimes[NodeID][]
           """     
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
    avgDefs = sum(Deficits)/len(Deficits)
    avgUndissem = sum(Undissem)/len(Undissem)
    avgMeanDelay = sum(MeanDelay)/len(MeanDelay)
    """
    Create a directory for these results and save them
    """
    Path(dirstr).mkdir(parents=True, exist_ok=True)
    np.savetxt(dirstr+'/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/avgTP.csv', avgTP, delimiter=',')
    np.savetxt(dirstr+'/avgWTP.csv', avgWTP, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLen.csv', avgInboxLen, delimiter=',')
    np.savetxt(dirstr+'/avgDefs.csv', avgDefs, delimiter=',')
    np.savetxt(dirstr+'/avgUndissem.csv', avgUndissem, delimiter=',')
    np.savetxt(dirstr+'/avgMeanDelay.csv', avgMeanDelay, delimiter=',')
    for NodeID in range(NUM_NODES):
        np.savetxt(dirstr+'/latTimes'+str(NodeID)+'.csv', np.asarray(latTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/latencies'+str(NodeID)+'.csv', np.asarray(latencies[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/interServTimes'+str(NodeID)+'.csv', np.asarray(interServTimes[NodeID]), delimiter=',')
        np.savetxt(dirstr+'/interArrTimes'+str(NodeID)+'.csv', np.asarray(interArrTimes[NodeID]), delimiter=',')
    nx.write_adjlist(G, dirstr+'/result_adjlist.txt', delimiter=' ')
    
def plot_results(dirstr):
    """
    Initialise plots
    """
    plt.close('all')
    
    """
    Load results from the data directory
    """
    avgLmds = np.loadtxt(dirstr+'/avgLmds.csv', delimiter=',')
    avgTP = np.loadtxt(dirstr+'/avgTP.csv', delimiter=',')
    avgInboxLen = np.loadtxt(dirstr+'/avgInboxLen.csv', delimiter=',')
    avgUndissem = np.loadtxt(dirstr+'/avgUndissem.csv', delimiter=',')
    avgMeanDelay = np.loadtxt(dirstr+'/avgMeanDelay.csv', delimiter=',')
    latencies = []
    ISTimes = []
    IATimes = []
    
    for NodeID in range(NUM_NODES):
        lat = [np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        if lat:
            latencies.append(lat)
        ist = [np.loadtxt(dirstr+'/interServTimes'+str(NodeID)+'.csv', delimiter=',')]
        if ist:
            ISTimes.append(ist)
        iat = [np.loadtxt(dirstr+'/interArrTimes'+str(NodeID)+'.csv', delimiter=',')]
        if iat:
            IATimes.append(iat)
    """
    Plot results
    """
    fig1, ax1 = plt.subplots(3,1, sharex=True, figsize=(8,8))
    ax1[0].title.set_text('Dissemination Rate')
    ax1[1].title.set_text('Scaled Dissemination Rate')
    ax1[2].title.set_text('# Undisseminated Transactions')
    ax1[0].grid(linestyle='--')
    ax1[1].grid(linestyle='--')
    ax1[2].grid(linestyle='--')
    ax1[2].set_xlabel('Time (sec)')
    #ax1[0].set_ylabel(r'${\lambda_i} / {\~{\lambda}_i}$')
    ax1[0].set_ylabel(r'$D_i$')
    ax1[1].set_ylabel(r'$D_i / {\~{\lambda}_i}$')    
    ax1[2].set_ylabel(r'$U_i$')
    plt.savefig(dirstr+'/Rates.png', bbox_inches='tight')
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue')
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:blue')
            ax1[2].plot(np.arange(10, SIM_TIME, STEP), avgUndissem[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue')
        if MODE[NodeID]==2:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red')
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:red')
            ax1[2].plot(np.arange(10, SIM_TIME, STEP), avgUndissem[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red')
        if MODE[NodeID]==3:
            ax1[0].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green')
            ax1[1].plot(np.arange(10, SIM_TIME, STEP), avgTP[1000:,NodeID]*sum(REP)/(NU*REP[NodeID]), linewidth=5*REP[NodeID]/REP[0], color='tab:green')
            ax1[2].plot(np.arange(10, SIM_TIME, STEP), avgUndissem[1000:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green')
    
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.grid(linestyle='--')
    ax2.set_xlabel('Time (sec)')
    ax2.plot(np.arange(10, SIM_TIME, STEP), np.sum(avgTP[1000:,:], axis=1), color = 'tab:green')
    ax22 = ax2.twinx()
    ax22.plot(np.arange(0, SIM_TIME, 1), avgMeanDelay, color='tab:gray')    
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax22.tick_params(axis='y', labelcolor='tab:gray')
    ax2.set_ylabel('Dissemination rate (Work/sec)', color='tab:green')
    ax22.set_ylabel('Mean Latency (sec)', color='tab:gray')
    fig2.tight_layout()
    plt.savefig(dirstr+'/Throughput.png', bbox_inches='tight')
    
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.grid(linestyle='--')
    ax3.set_xlabel('Latency (sec)')
    plot_cdf(latencies, ax3)
    plt.savefig(dirstr+'/Latency.png', bbox_inches='tight')
    
    fig4, ax4 = plt.subplots(figsize=(8,4))
    ax4.set_xlabel('Time (sec)')
    ax4.set_ylabel(r'$\lambda_i$')
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:blue')
        if MODE[NodeID]==2:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:red')
        if MODE[NodeID]==3:
            ax4.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,NodeID], linewidth=5*REP[NodeID]/REP[0], color='tab:green')
    
    fig5, ax5 = plt.subplots(figsize=(8,4))
    ax5.set_xlabel('Time (sec)')
    ax5.set_ylabel('Own Inbox')
    for NodeID in range(NUM_NODES):
        if MODE[NodeID]==1:
            ax5.plot(np.arange(0, SIM_TIME, STEP), avgInboxLen[:,NodeID], color='tab:blue')
        if MODE[NodeID]==2:
            ax5.plot(np.arange(0, SIM_TIME, STEP), avgInboxLen[:,NodeID], color='tab:red')
    
    fig6, ax6 = plt.subplots(figsize=(8,4))
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
        if MODE[NodeID]==3:
            ax6.bar(NodeID, REP[NodeID], color='tab:green')
    
    fig7, ax7 = plt.subplots(figsize=(8,4))
    plot_cdf(ISTimes, ax7)
    ax7.grid(linestyle='--')
    ax7.set_xlabel('Inter-service time (sec)')
    plt.savefig(dirstr+'/InterServiceTimes.png', bbox_inches='tight')
    
    fig8, ax8 = plt.subplots(figsize=(8,4))
    plot_cdf_exp(IATimes, ax8)
    ax8.grid(linestyle='--')
    ax8.set_xlabel('Inter-arrival time (sec)')
    plt.savefig(dirstr+'/InterArrivalTimes.png', bbox_inches='tight')
    
def plot_cdf(data, ax):
    step = STEP/10
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(data[NodeID][0])>0:
            val = np.max(data[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    Lines = [[] for NodeID in range(NUM_NODES)]
    for NodeID in range(len(data)):
        if MODE[NodeID]>0:
            bins = np.arange(0, round(maxval*1/step), 1)*step
            pdf = np.zeros(len(bins))
            for i, b in enumerate(bins):
                lats = [lat for lat in data[NodeID][0] if (lat>b and lat<b+step)]
                pdf[i] = len(lats)
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            if MODE[NodeID]==1:
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:blue', linewidth=4*REP[NodeID]/REP[0])
            if MODE[NodeID]==2:
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:red', linewidth=4*REP[NodeID]/REP[0])
            if MODE[NodeID]==3:
                Lines[NodeID] = ax.plot(bins, cdf, color='tab:green', linewidth=4*REP[NodeID]/REP[0])
    ModeLines = [Line2D([0],[0],color='tab:blue', lw=4), Line2D([0],[0],color='tab:red', lw=4)]
    ax.legend(ModeLines, ['Content','Best-effort'], loc='lower right')
    
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
            for i, b in enumerate(bins):
                lats = [lat for lat in data[NodeID][0] if (lat>b and lat<b+step)]
                pdf[i] = len(lats)
            pdf = pdf/sum(pdf) # normalise
            cdf = np.cumsum(pdf)
            ax.plot(bins, cdf, color='tab:red')
    ax.axvline(np.mean(data[0][0]), linestyle='--', color='tab:red')

    ax.plot(bins, np.ones(len(bins))-np.exp(-NU*bins), color='black')
    ax.axvline(1/NU, linestyle='--', color='black')
    #ax.plot(bins, np.ones(len(bins))-np.exp(-0.95*NU*bins), linestyle='--', color='tab:red')
    ModeLines = [Line2D([0],[0],color='tab:red', lw=2), Line2D([0],[0],linestyle='--',color='black', lw=2)]
    ax.legend(ModeLines, ['Measured',r'$1-e^{-\nu t}$'], loc='lower right')

class Transaction:
    """
    Object to simulate a transaction its edges in the DAG
    """
    def __init__(self, IssueTime, Parents, Node, Work=0, Index=None):
        self.IssueTime = IssueTime
        self.Children = []
        self.Parents = Parents
        self.Index = Index
        self.InformedNodes = 0
        self.GlobalSolidTime = []
        self.Work = Work
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
        else: # genesis
            self.NodeID = []

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Ledger = [Genesis]
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox()
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
        self.ServiceTimes = [[] for NodeID in range(NUM_NODES)]
        self.ArrivalTimes = []
        
    def issue_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2:
                if self.BackOff:
                    self.LastIssueTime += BETA*REP[self.NodeID]/self.Lambda
                while Time+STEP >= self.LastIssueTime + self.LastIssueWork/self.Lambda:
                    self.LastIssueTime += self.LastIssueWork/self.Lambda
                    Parents = self.select_tips()
                    Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.LastIssueWork = Work
                    self.IssuedTrans.append(Transaction(self.LastIssueTime, Parents, self, Work))
            else:
                times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda)))
                for t in times:
                    Parents = self.select_tips()
                    Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.IssuedTrans.append(Transaction(t, Parents, self, Work))
        
        # check PoW completion
        while self.IssuedTrans:
            Tran = self.IssuedTrans.pop(0)
            p = Packet(self, self, Tran, Tran.IssueTime)
            p.EndTime = Tran.IssueTime
            if MODE[self.NodeID]==3: # malicious don't consider own txs for scheduling
                self.add_to_ledger(self, Tran, Tran.IssueTime)
            else:
                self.add_to_inbox(p, Tran.IssueTime)
    
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
        while self.LastScheduleTime+(self.LastScheduleWork/NU)<Time+STEP:
            if SCHEDULING=='drr':
                Packet = self.Inbox.drr_schedule()
            elif SCHEDULING=='drr_lds':
                Packet = self.Inbox.drr_lds_schedule()
            elif SCHEDULING=='drrpp':
                Packet = self.Inbox.drrpp_schedule()
            elif SCHEDULING=='fifo':
                Packet = self.Inbox.fifo_schedule(Time)
            if Packet is not None:
                if Packet.Data not in self.Ledger:
                    self.add_to_ledger(Packet.TxNode, Packet.Data, max(Time, self.LastScheduleTime+(1/NU)))
                # update AIMD
                if Packet.Data.NodeID==self.NodeID:
                    self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*len(self.Inbox.Packets[self.NodeID])
                self.set_rate(Time)
                self.LastScheduleTime = max(Time, self.LastScheduleTime+(self.LastScheduleWork/NU))
                self.LastScheduleWork = Packet.Data.Work
                self.ServiceTimes[Packet.Data.NodeID].append(Time)
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
        # mark this TX as received by this node
        Tran.InformedNodes += 1
        if Tran.InformedNodes==NUM_NODES:
            self.Network.Throughput[Tran.NodeID] += 1
            self.Network.WorkThroughput[Tran.NodeID] += Tran.Work
            self.Network.TranDelays.append(Time-Tran.IssueTime)
            self.Network.DissemTimes.append(Time)
            Tran.GlobalSolidTime = Time
            self.Network.Nodes[Tran.NodeID].Undissem -= 1
            self.Network.Nodes[Tran.NodeID].UndissemWork -= Tran.Work
            
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
            if MODE[self.NodeID]==2 and Time>=START_TIMES[self.NodeID]: # AIMD starts after 1 min adjustment
                # if wait time has not passed---reset.
                if self.LastBackOff:
                    if Time < self.LastBackOff + BETA*REP[self.NodeID]/self.Lambda:
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
        if Packet.Data not in self.Inbox.Trans:
            if Packet.Data not in self.Ledger:
                self.Inbox.AllPackets.append(Packet)
                NodeID = Packet.Data.NodeID
                if NodeID in self.Inbox.Empty:
                    self.Inbox.Empty.remove(NodeID)
                    self.Inbox.New.append(NodeID)
                    self.Inbox.Deficit[NodeID] += QUANTUM[NodeID]
                self.Inbox.Packets[NodeID].append(Packet)
                self.Inbox.Trans.append(Packet.Data)
                self.ArrivalTimes.append(Time)
                if NodeID==self.NodeID:
                    self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*len(self.Inbox.Packets[self.NodeID])
                    self.check_congestion(Time)
                
                
class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self):
        self.AllPackets = [] # Inbox_m
        self.Packets = [] # Inbox_m(i)
        for NodeID in range(NUM_NODES):
            self.Packets.append([])
        self.Trans = []
        self.RRNodeID = np.random.randint(NUM_NODES) # start at a random node
        self.Deficit = np.zeros(NUM_NODES)
        self.Scheduled = []
        self.Avg = 0
        self.New = []
        self.Old = []
        self.Empty = [NodeID for NodeID in range(NUM_NODES)]
       
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
        if self.Scheduled:
            return self.Scheduled.pop(0)
        if self.AllPackets:
            while not self.Scheduled:
                if self.Packets[self.RRNodeID]:
                    self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
                    while self.Packets[self.RRNodeID]:
                        Work = self.Packets[self.RRNodeID][0].Data.Work
                        if self.Deficit[self.RRNodeID]>=Work:
                            Packet = self.Packets[self.RRNodeID][0]
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
    
    def drr_lds_schedule(self):
        if self.Scheduled:
            return self.Scheduled.pop(0)
        if self.AllPackets:
            while not self.Scheduled:
                if self.Packets[self.RRNodeID]:
                    self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
                else:
                    self.Deficit[self.RRNodeID] = min(self.Deficit[self.RRNodeID]+QUANTUM[self.RRNodeID], MAX_WORK)
                while self.Packets[self.RRNodeID]:
                    Work = self.Packets[self.RRNodeID][0].Data.Work
                    if self.Deficit[self.RRNodeID]>=Work:
                        Packet = self.Packets[self.RRNodeID][0]
                        self.Deficit[self.RRNodeID] -= Work
                        # remove the transaction from all inboxes
                        self.remove_packet(Packet)
                        self.Scheduled.append(Packet)
                    else:
                        break
                self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
            return self.Scheduled.pop(0)
        
    def drrpp_schedule(self):
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
                    
    def fifo_schedule(self, Time):
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
        self.WorkThroughput = [0 for NodeID in range(NUM_NODES)]
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
            Node.issue_txs(Time)
            Node.schedule_txs(Time)
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time+STEP)
    
    def tran_latency(self, latencies, latTimes):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.IssueTime>20:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.IssueTime)
                latTimes[Tran.NodeID].append(Tran.GlobalSolidTime)
        return latencies, latTimes
                
if __name__ == "__main__":
        main()
        
    
    