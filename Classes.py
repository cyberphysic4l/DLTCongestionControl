# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
from pathlib import Path

# Simulation Parameters
MONTE_CARLOS = 1
SIM_TIME = 600
SETTLING_TIME = 100
STEP = 0.1
# Network Parameters
NU = 10
NUM_NODES = 10
NUM_NEIGHBOURS = 4
REP = np.ones(NUM_NODES)
REP[0] = 5
REP[1] = 5
MAX_PRIORITY = 20*sum(REP)
MIN_PRIORITY = -MAX_PRIORITY
    
def main():
    dirstr = 'data/priorityinboxnoaimd/nu='+str(NU)+'/'+'nodes='+str(NUM_NODES)+'/'+'neighbours='+str(NUM_NEIGHBOURS)+'/'+'rep='+''.join(str(int(e)) for e in REP)+'/'+'simtime='+str(SIM_TIME)+'/'+'nmc='+str(MONTE_CARLOS)
    if not Path(dirstr).exists():
        print("Simulating")
        simulate(dirstr)
    else:
        print("Simulation already done for these parameters")
        simulate(dirstr)
    plot_results(dirstr)
    
def simulate(dirstr):
    """
    Setup simulation inputs and instantiate output arrays
    """
    # seed rng
    np.random.seed(0)
    TimeSteps = int(SIM_TIME/STEP)
    """
    Generate network topology:
    Comment out one of the below lines for either random k-regular graph or a
    graph from an adjlist txt file i.e. from the autopeering simulator
    """
    G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES)
    #G = nx.read_adjlist('input_adjlist.txt', delimiter=' ')
    # Get adjacency matrix and weight by delay at each channel
    ChannelDelays = 0.9*np.ones((NUM_NODES, NUM_NODES))+0.2*np.random.rand(NUM_NODES, NUM_NODES)
    AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
    # Node parameters
    MaxLambdas = NU*REP/sum(REP)
    Nus = NU*(np.ones(NUM_NODES))
    """
    Monte Carlo Sims
    """
    Lmds = []
    InboxLens = []
    SymDiffs = []
    latencies = []
    for NodeID in range(NUM_NODES):
        latencies.append([])
    for mc in range(MONTE_CARLOS):
        print(mc)
        Net = Network(AdjMatrix, MaxLambdas, Nus)
        Lmds.append(np.zeros((TimeSteps, NUM_NODES)))
        InboxLens.append(np.zeros((TimeSteps, NUM_NODES)))
        SymDiffs.append(np.zeros(TimeSteps))
        for i in range(TimeSteps):
            # discrete time step size specified by global variable STEP
            T = STEP*i
            # update network for all new events in this time step
            Net.simulate(T)
            # save summary results in output arrays
            for Node in Net.Nodes:
                if len(Node.IssueTimes)>20:
                    Lmds[mc][i, Node.NodeID] = 20/(Node.IssueTimes[-1]-Node.IssueTimes[-20])
                InboxLens[mc][i, Node.NodeID] = len(Net.Nodes[0].Inbox.Packets[Node.NodeID])
            SymDiffs[mc][i] = Net.sym_diffs().max()
        latencies = Net.tran_latency(latencies)
        for NodeID in range(NUM_NODES):
            print(Net.Nodes[NodeID].EmptyInbox)
        del Net 
    """
    Get results
    """
    avgLmds = sum(Lmds)/len(Lmds)
    avgUtil = 100*np.sum(avgLmds, axis=1)/NU
    avgInboxLen = sum(InboxLens)/len(InboxLens)
    avgMSDs = sum(SymDiffs)/len(SymDiffs)
    """
    Create a directory for these results and save them
    """
    Path(dirstr).mkdir(parents=True, exist_ok=True)
    np.savetxt(dirstr+'/avgMSDs.csv', avgMSDs, delimiter=',')
    np.savetxt(dirstr+'/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/avgUtil.csv', avgUtil, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLen.csv', avgInboxLen, delimiter=',')
    for NodeID in range(NUM_NODES):
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
    latencies = []
    for NodeID in range(NUM_NODES):
        latencies.append([np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')])
    """
    Plot results
    """
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda')
    ax1.legend(list(map(str, range(NUM_NODES))))
    for i in range(NUM_NODES):
        ax1.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,i])
    for i in range(NUM_NODES):
        ax1.plot([0, SIM_TIME], [REP[NodeID]*NU/sum(REP), REP[NodeID]*NU/sum(REP)], 'r--')
    plt.legend(range(NUM_NODES))
    plt.savefig(dirstr+'/Lambdas.png')
    
    fig2, ax2 = plt.subplots()
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Max Symmetric Difference')
    ax2.plot(np.arange(0, SIM_TIME, STEP), avgMSDs)
    plt.savefig(dirstr+'/SymDif.png')
    
    fig3, ax3 = plt.subplots()
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Network Utilisation (%)')
    ax3.plot(np.arange(0, SIM_TIME, STEP), avgUtil)
    ax3.plot([0, SIM_TIME], [100, 100], 'r--')
    plt.savefig(dirstr+'/Util.png')
    
    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Inbox Length')
    ax4.plot(np.arange(0, SIM_TIME, STEP), avgInboxLen)
    plt.legend(range(NUM_NODES))
    plt.savefig(dirstr+'/Inbox.png')
    
    fig5, ax5 = plt.subplots(NUM_NODES, sharex=True, sharey=True)
    maxval = 0
    for NodeID in range(NUM_NODES):
        val = np.max(latencies[NodeID])
        if val>maxval:
            maxval = val
    for NodeID in range(NUM_NODES):
        bins = np.arange(0, round(maxval), STEP)
        ax5[NodeID].hist(latencies[NodeID], bins=bins, density=True)
    plt.savefig(dirstr+'/Latency.png')
    """
    Draw network graph used in this simulation
    """
    G = nx.read_adjlist(dirstr+'/result_adjlist.txt', delimiter=' ')
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos)#, node_color=colors[0:NUM_NODES])
    plt.show()

class Transaction:
    """
    Object to simulate a transaction its edges in the DAG
    """
    def __init__(self, ArrivalTime, Parents, Node):
        self.ArrivalTime = ArrivalTime
        self.Children = []
        self.Parents = Parents
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
            self.InformedNodes = [Node] # info about who has seen this TX
            self.GlobalSolidTime = []
        else: # genesis
            self.NodeID = []
            self.InformedNodes = [] 
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
    def __init__(self, Network, MaxLambda, Nu, NodeID, Genesis, PoWDelay = STEP):
        self.TipsSet = [Genesis]
        self.Ledger = [Genesis]
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox()
        self.Inbox.Active[NodeID] = True
        self.Inbox.ActiveRep = REP[NodeID]
        self.Nu = Nu
        self.MaxLambda = MaxLambda
        self.NodeID = NodeID
        self.IssueTimes = []
        self.EmptyInbox = 0
                    
    def issue_tx(self, Time):
        """
        Take a transaction from 'buffer', add it to Ledger and broadcast it
        """
        self.IssueTimes.append(Time)
        Parents = self.select_tips()
        Tran = Transaction(Time, Parents, self)
        p = Packet(self, self, Tran, Time)
        p.EndTime = Time
        self.add_to_ledger(self, Tran, Time)
    
    def select_tips(self):
        """
        Implements uniform random tip selection
        """
        if len(self.TipsSet)>1:
            Selection = sample(self.TipsSet, 2)
        else:
            Selection = self.Ledger[-2:-1]
        return Selection
    
    def add_to_ledger(self, TxNode, Tran, Time):
        """
        Adds the transaction to the local copy of the ledger and broadcast it
        """
        self.Ledger.append(Tran)
        # mark this TX as received by this node
        Tran.InformedNodes.append(self)
        if len(Tran.InformedNodes)==NUM_NODES:
            Tran.GlobalSolidTime = Time
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
    
    def process_inbox(self, Time):
        """
        Process TXs in inbox of TXs received from neighbours
        """
        # sort inboxes by arrival time
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        nTX = np.random.poisson(STEP*self.Nu)
        times = np.sort(np.random.uniform(Time, Time+STEP, nTX))
        i = 0
        while i < nTX:
            if self.Inbox.AllPackets:
                # update priority of inbox channels (assume all active for now)
                for NodeID in range(NUM_NODES):
                    if self.Inbox.Packets[NodeID] and not self.Inbox.Active[NodeID]:
                        self.Inbox.Active[NodeID] = True
                        self.Inbox.ActiveRep += REP[NodeID]
                    if self.Inbox.Active[NodeID]:
                        self.Inbox.Priority[NodeID] += REP[NodeID]
                # First sort by priority
                PriorityOrder = sorted(range(NUM_NODES), key=lambda k: self.Inbox.Priority[k], reverse=True)
                # take highest priority nonempty queue with oldest tx
                EarliestTime = Time
                HighestPriority = -float('Inf')
                for NodeID in PriorityOrder:
                    Priority = self.Inbox.Priority[NodeID]
                    if NodeID==self.NodeID:
                        if Priority>HighestPriority:
                            HighestPriority = Priority
                            BestNodeID = NodeID
                    elif self.Inbox.Packets[NodeID]:
                        if Priority>=HighestPriority:
                            HighestPriority = Priority
                            ArrivalTime = self.Inbox.Packets[NodeID][0].EndTime
                            if ArrivalTime<=EarliestTime:
                                EarliestTime = ArrivalTime
                                BestNodeID = NodeID
                        else:
                            break
                if BestNodeID==self.NodeID:
                    self.issue_tx(times[i])
                else:
                    Tran = self.Inbox.Packets[BestNodeID][0].Data
                    TxNode = self.Inbox.Packets[BestNodeID][0].TxNode
                    # add this to ledger if it is not already there
                    if Tran not in self.Ledger:
                        self.add_to_ledger(TxNode, Tran, times[i])
                    # remove the transaction from all inboxes
                    self.remove_from_inbox(self.Inbox.Packets[BestNodeID][0])
                # reduce priority of the inbox channel by total active rep amount
                self.Inbox.Priority[BestNodeID] -= self.Inbox.ActiveRep
                i += 1
            else:
                if Time>200:
                    self.EmptyInbox += 1
                
                if np.random.random()<0.5*REP[self.NodeID]/self.Inbox.ActiveRep:
                    # update priority of inbox channels (assume all active for now)
                    for NodeID in range(NUM_NODES):
                        if self.Inbox.Active[NodeID]:
                            self.Inbox.Priority[NodeID] += REP[NodeID]
                    BestNodeID = self.NodeID
                    self.issue_tx(times[i])
                    # reduce priority of the inbox channel by total active rep amount
                    self.Inbox.Priority[BestNodeID] -= self.Inbox.ActiveRep
                i += 1
            
    def add_to_inbox(self, Packet):
        """
        Add to inbox if not already received and/or processed
        """
        if Packet.Data not in self.Inbox.Trans:
            if Packet.Data not in self.Ledger:
                self.Inbox.AllPackets.append(Packet)
                self.Inbox.Packets[Packet.Data.NodeID].append(Packet)
                self.Inbox.Trans.append(Packet.Data)
        
    def remove_from_inbox(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.Inbox.Trans:
            if Packet in self.Inbox.AllPackets:
                self.Inbox.AllPackets.remove(Packet)
                self.Inbox.Packets[Packet.Data.NodeID].remove(Packet)
                self.Inbox.Trans.remove(Packet.Data)
                
class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self):
        self.AllPackets = []
        self.Packets = []
        for NodeID in range(NUM_NODES):
            self.Packets.append([])
        self.Trans = []
        self.Priority = np.zeros(NUM_NODES)
        self.Active = np.zeros(NUM_NODES, dtype=bool)
        self.ActiveRep = 0

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
                t = Packet.StartTime+self.Delay
                if(t<=Time):
                    self.deliver_packet(Packet, t)
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        """
        When packet has arrived at receiving node, process it
        """
        Packet.EndTime = Time
        if isinstance(Packet.Data, Transaction):
            # if this is a transaction, add the Packet to Inbox
            self.RxNode.add_to_inbox(Packet)
        else: 
            # else this is a back off notification
            self.RxNode.process_cong_notif(Packet, Time)
        self.Packets.remove(Packet)
        
class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix, MaxLambdas, Nus):
        self.A = AdjMatrix
        self.MaxLambdas = MaxLambdas
        self.Nodes = []
        self.CommChannels = []
        Genesis = Transaction(0, [], [])
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(Node(self, MaxLambdas[i], Nus[i], i, Genesis))
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
        Each node process inboxes which triggers transaction issuing
        """
        for Node in self.Nodes:
            Node.process_inbox(Time)
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time)
    
    def sym_diffs(self):
        SymDiffs = np.zeros((NUM_NODES, NUM_NODES))
        for i, iNode in enumerate(self.Nodes):
            for j, jNode in enumerate(self.Nodes):
                if j>i:
                    SymDiffs[i][j] = len(set(iNode.Ledger).symmetric_difference(set(jNode.Ledger)))
        return SymDiffs + np.transpose(SymDiffs)
    
    def tran_latency(self, latencies):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.ArrivalTime>200:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.ArrivalTime)
        return latencies
                
if __name__ == "__main__":
        main()
        
    
    