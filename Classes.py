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
MONTE_CARLOS = 10
SIM_TIME = 600
STEP = 0.1
# Network Parameters
NU = 10
NUM_NODES = 10
NUM_NEIGHBOURS = 4
MANA = np.ones(NUM_NODES)
MANA[0] = 5
MANA[1] = 5
# AIMD Parameters
ALPHA = 0.01*NU
BETA = 0.7
WAIT_TIME = 5
MAX_INBOX_LEN = NU # length of inbox that would empty in WAIT_TIME
    
def main():
    dirstr = 'data/priorityinbox/samegraph/nu='+str(NU)+'/'+'alpha='+str(ALPHA)+'/'+'beta='+str(BETA)+'/'+'tau='+str(WAIT_TIME)+'/'+'inbox='+str(MAX_INBOX_LEN)+'/'+'nodes='+str(NUM_NODES)+'/'+'neighbours='+str(NUM_NEIGHBOURS)+'/'+'mana='+''.join(str(int(e)) for e in MANA)+'/'+'simtime='+str(SIM_TIME)+'/'+'nmc='+str(MONTE_CARLOS)
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
    Lambdas = NU*MANA/sum(MANA)
    Nus = NU*(np.ones(NUM_NODES))
    """
    Monte Carlo Sims
    """
    mcResults = [[],[],[],[]]
    latencies = []
    for mc in range(MONTE_CARLOS):
        print(mc)
        
        Net = Network(AdjMatrix, Lambdas, Nus)
        mcResults[0].append(np.zeros((TimeSteps, NUM_NODES)))
        mcResults[1].append(np.zeros((TimeSteps, NUM_NODES)))
        mcResults[2].append(np.zeros(TimeSteps))
        for i in range(TimeSteps):
            # discrete time step size specified by global variable STEP
            T = STEP*i
            # update network for all new events in this time step
            Net.Nodes[0].Lambda = 0
            Net.simulate(T)
            # save summary results in output arrays
            for Node in Net.Nodes:
                mcResults[0][mc][i, Node.NodeID] = Node.Lambda
                mcResults[1][mc][i, Node.NodeID] = len(Node.FilteredInbox)
            SymDiffs = Net.sym_diffs()
            mcResults[2][mc][i] = SymDiffs.max()
        latencies += Net.tran_latency()
        del Net
    """
    Get results
    """
    avgLmds = sum(mcResults[0])/len(mcResults[0])
    avgUtil = 100*np.sum(avgLmds, axis=1)/NU
    avgInboxLen = sum(mcResults[1])/len(mcResults[1])
    avgMSDs = sum(mcResults[2])/len(mcResults[2])
    """
    Create a directory for these results and save them
    """
    Path(dirstr).mkdir(parents=True, exist_ok=True)
    np.savetxt(dirstr+'/avgMSDs.csv', avgMSDs, delimiter=',')
    np.savetxt(dirstr+'/avgLmds.csv', avgLmds, delimiter=',')
    np.savetxt(dirstr+'/avgUtil.csv', avgUtil, delimiter=',')
    np.savetxt(dirstr+'/avgInboxLen.csv', avgInboxLen, delimiter=',')
    np.savetxt(dirstr+'/latencies.csv', np.asarray(latencies), delimiter=',')
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
    latencies = np.loadtxt(dirstr+'/latencies.csv', delimiter=',')
    """
    Plot results
    """
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Lambda')
    ax1.legend(list(map(str, range(NUM_NODES))))
    for i in range(NUM_NODES):
        ax1.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,i])
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
    plt.savefig(dirstr+'/Util.png')
    
    fig4, ax4 = plt.subplots()
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Inbox Length')
    ax4.plot(np.arange(0, SIM_TIME, STEP), avgInboxLen)
    plt.savefig(dirstr+'/Inbox.png')
    
    plt.figure()
    bins = np.arange(0, round(max(latencies)), STEP)
    plt.hist(latencies, bins=bins, density=True)
    plt.xlabel('Latency(s)')
    plt.ylabel('Probability')
    plt.savefig(dirstr+'/Latency.png')
    """
    Draw network graph used in this simulation
    """
    G = nx.read_adjlist(dirstr+'/result_adjlist.txt', delimiter=' ')
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos)#, node_color=colors[0:NUM_NODES])
    plt.show()

def mana_length(Inbox):
    ManaLength = 0
    if Inbox:
        for Packet in Inbox:
            NodeID = Packet.Data.NodeID
            ManaLength += 1/MANA[NodeID]
        return ManaLength
    else:
        return 0

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
    def __init__(self, Network, Lambda, Nu, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Tangle = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = []
        self.InboxTrans = []
        self.FilteredInbox = []
        self.Lambda = Lambda
        self.Nu = Nu
        self.NodeID = NodeID
        self.Alpha = ALPHA*MANA[self.NodeID]/sum(MANA)
        self.BackOff = []
        self.LastBackOff = []
        self.LastCongestion = []
        self.CongNotifRx = False
        
        self.PriorityInbox = []
        for i in range(NUM_NODES):
            self.PriorityInbox.append([])
        
    def generate_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        NewTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda)))
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
                    p.EndTime = t
                    self.add_to_inbox(p)
    
    def select_tips(self):
        """
        Implements uniform random tip selection
        """
        if len(self.TipsSet)>1:
            Selection = sample(self.TipsSet, 2)
        else:
            Selection = self.Tangle[-2:-1]
        return Selection
    
    def process_inbox(self, Time):
        """
        Process TXs in inbox of TXs received from neighbours
        """
        # sort inboxes by arrival time
        for NodeID in range(NUM_NODES):
            self.PriorityInbox[NodeID].sort(key=lambda p: p.EndTime)
        self.FilteredInbox.sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        nTX = np.random.poisson(STEP*self.Nu)
        times = np.sort(np.random.uniform(Time, Time+STEP, nTX))
        i = 0
        while i < nTX:
            if self.FilteredInbox:
                # calculate current active mana from inbox contents
                ActiveMana = 0
                for NodeID in range(NUM_NODES):
                    if self.PriorityInbox[NodeID]:
                        ActiveMana += MANA[NodeID]
                # update priority of inbox packets
                for Packet in self.FilteredInbox:
                    IssuingNodeID = Packet.Data.NodeID
                    PInboxLen = len(self.PriorityInbox[IssuingNodeID])
                    Packet.Priority += MANA[IssuingNodeID]/(ActiveMana*PInboxLen)
                # sort filtered inbox by priority
                self.FilteredInbox.sort(key=lambda p: p.Priority)
                # take highest priority transaction
                Tran = self.FilteredInbox[0].Data
                if Tran not in self.Tangle:
                    self.Tangle.append(Tran)
                    # mark this TX as received by this node
                    Tran.InformedNodes.append(self)
                    if len(Tran.InformedNodes)==NUM_NODES:
                        Tran.GlobalSolidTime = times[i]
                    if not Tran.Children:
                        self.TipsSet.append(Tran)
                    if Tran.Parents:
                        for Parent in Tran.Parents:
                            Parent.Children.append(Tran)
                            if Parent in self.TipsSet:
                                self.TipsSet.remove(Parent)
                            else:
                                continue
                # reset priority of the packet for forwarding
                self.FilteredInbox[0].Priority = 0
                # broadcast the packet
                self.Network.broadcast_data(self, self.FilteredInbox[0], times[i])
                # remove the transaction from all inboxes
                self.remove_from_inbox(self.FilteredInbox[0])
                i += 1
            else:
                break
    
    def check_congestion(self, Time):
        """
        Check if congestion is occurring
        """
        if (len(self.FilteredInbox) > MAX_INBOX_LEN):
            # ignore it if recently backed off
            if self.LastCongestion:
                if Time < self.LastCongestion + WAIT_TIME:
                    return
                
            self.LastCongestion = Time
            # back off if occupying more than entitled share of the Inbox
            Utils = []
            for NodeID in range(NUM_NODES):
                if self.PriorityInbox[NodeID]:
                    Utils.append(len(self.PriorityInbox[NodeID])/MANA[NodeID])
                
            if len(self.PriorityInbox[self.NodeID])/MANA[self.NodeID]>np.mean(Utils):
                self.BackOff = True
            
    def aimd_update(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        # if wait time has not passed---reset.
        if self.LastBackOff:
            if Time < self.LastBackOff + WAIT_TIME:
                self.BackOff = False
                return
        # multiplicative decrease or else additive increase
        if self.BackOff:
            self.Lambda = self.Lambda*BETA
            self.LastBackOff = Time
        else:
            self.Lambda += self.Alpha*STEP
            
    def add_to_inbox(self, Packet):
        """
        Add to inbox if not already received and/or processed
        """
        if Packet.Data not in self.InboxTrans:
            if Packet.Data not in self.Tangle:
                self.FilteredInbox.append(Packet)
                IssuingNodeID = Packet.Data.NodeID
                self.PriorityInbox[IssuingNodeID].append(Packet)
                self.InboxTrans.append(Packet.Data)
        
    def remove_from_inbox(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.FilteredInbox:
            if Packet in self.FilteredInbox:
                self.FilteredInbox.remove(Packet)
                IssuingNodeID = Packet.Data.NodeID
                self.PriorityInbox[IssuingNodeID].remove(Packet)
                self.InboxTrans.remove(Packet.Data)

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
        self.Priority = 0
        
    def increment_priority(self):
        self.Priority += MANA[self.Data.NodeID]
    
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
    def __init__(self, AdjMatrix, Lambdas, Nus):
        self.A = AdjMatrix
        self.Lambdas = Lambdas
        self.Nodes = []
        self.CommChannels = []
        Genesis = Transaction(0, [], [])
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(Node(self, Lambdas[i], Nus[i], i, Genesis))
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
            
    def broadcast_data(self, TxNode, Packet, Time):
        """
        Send this data (TX or back off) to all neighbours
        """
        for i, CC in enumerate(self.CommChannels[self.Nodes.index(TxNode)]):
            # do not send to this node if it was received from this node
            if isinstance(Packet.Data, Transaction):
                if Packet.TxNode==TxNode.Neighbours[i]:
                    continue
            CC.send_packet(TxNode, TxNode.Neighbours[i], Packet.Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate and process new transactions
        """
        for Node in self.Nodes:
            Node.generate_txs(Time)
            Node.process_inbox(Time)
            Node.check_congestion(Time)
            Node.aimd_update(Time)
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
                    SymDiffs[i][j] = len(set(iNode.Tangle).symmetric_difference(set(jNode.Tangle)))
        return SymDiffs + np.transpose(SymDiffs)
    
    def tran_latency(self):
        latencies = []
        for Tran in self.Nodes[0].Tangle:
            if Tran.GlobalSolidTime:
                latencies.append(Tran.GlobalSolidTime-Tran.ArrivalTime)
        return latencies
                
if __name__ == "__main__":
        main()
        
    
    