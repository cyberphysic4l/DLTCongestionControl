# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from random import sample
from pathlib import Path
import math

# Simulation Parameters
MONTE_CARLOS = 100
SIM_TIME = 1200
STEP = 0.1
# Network Parameters
NU = 10
NUM_NODES = 10
NUM_NEIGHBOURS = 4
REP = np.ones(NUM_NODES)
REP[0] = 5
REP[1] = 5
MAX_BURST = 3*sum(REP)
# AIMD Parameters
ALPHA = 0.002*NU
BETA = 0.8
WAIT_TIME = 20
MAX_INBOX_LEN = 5 # worst case number of TXs that empty each second

SCHEDULING = 'gps'
TP = 3
    
def main():
    dirstr = 'data/scheduling='+SCHEDULING+'/tokenbucket'+'/nu='+str(NU)+'/'+'alpha='+str(ALPHA)+'/'+'beta='+str(BETA)+'/'+'tau='+str(WAIT_TIME)+'/'+'inbox='+str(MAX_INBOX_LEN)+'/'+'nodes='+str(NUM_NODES)+'/'+'neighbours='+str(NUM_NEIGHBOURS)+'/'+'rep='+''.join(str(int(e)) for e in REP)+'/'+'simtime='+str(SIM_TIME)+'/'+'nmc='+str(MONTE_CARLOS)
    if not Path(dirstr).exists():
        print("Simulating")
        simulate(dirstr)
    else:
        print("Simulation already done for these parameters")
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
    SymDiffs = []
    latencies = []
    for NodeID in range(NUM_NODES):
        latencies.append([])
    for mc in range(MONTE_CARLOS):
        print(mc)
        """
        Generate network topology:
        Comment out one of the below lines for either random k-regular graph or a
        graph from an adjlist txt file i.e. from the autopeering simulator
        """
        G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES)
        #G = nx.read_adjlist('input_adjlist.txt', delimiter=' ')
        # Get adjacency matrix and weight by delay at each channel
        ChannelDelays = 0.09*np.ones((NUM_NODES, NUM_NODES))+0.02*np.random.rand(NUM_NODES, NUM_NODES)
        AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
        # Node parameters
        Lambdas = NU*REP/sum(REP)
        Nus = NU*(np.ones(NUM_NODES))
        Net = Network(AdjMatrix, Lambdas, Nus)
        Lmds.append(np.zeros((TimeSteps, NUM_NODES)))
        InboxLens.append(np.zeros((TimeSteps, NUM_NODES)))
        SymDiffs.append(np.zeros(TimeSteps))
        for i in range(TimeSteps):
            # discrete time step size specified by global variable STEP
            T = STEP*i
            # update network for all new events in this time step
            Net.Nodes[0].Lambda = 0
            Net.simulate(T)
            # save summary results in output arrays
            for Node in Net.Nodes:
                Lmds[mc][i, Node.NodeID] = Node.Lambda
                InboxLens[mc][i, Node.NodeID] = len(Net.Nodes[0].Inbox.Packets[Node.NodeID])
            SymDiffs[mc][i] = Net.sym_diffs().max()
        latencies = Net.tran_latency(latencies)
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
    plot_results(dirstr)
        
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
        lat = [np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        if lat:
            latencies.append(lat)
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
        ax1.plot([0, SIM_TIME], [REP[i]*NU/sum(REP), REP[i]*NU/sum(REP)], 'r--')
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
    
    fig5, ax5 = plt.subplots()
    maxval = 0
    for NodeID in range(NUM_NODES):
        if len(latencies[NodeID][0])>0:
            val = np.max(latencies[NodeID][0])
        else:
            val = 0
        if val>maxval:
            maxval = val
    
    for NodeID in range(len(latencies)):
        bins = np.arange(0, round(maxval), STEP)
        pdf = np.zeros(len(bins))
        for i, b in enumerate(bins):
            lats = [lat for lat in latencies[NodeID][0] if (lat>b and lat<b+STEP)]
            pdf[i] = len(lats)
        pdf = pdf/sum(pdf) # normalise
        cdf = np.cumsum(pdf)
        ax5.plot(bins, cdf)
    plt.legend(range(NUM_NODES))
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
        self.IssueTime = []
        self.Children = []
        self.Parents = Parents
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
            self.InformedNodes = [Node] # info about who has seen this TX
            self.GlobalSolidTime = []
            self.GlobalSolidTime90pc = []
        else: # genesis
            self.NodeID = []
            self.InformedNodes = [] 
            self.GlobalSolidTime = []
            self.GlobalSolidTime90pc = []
        
        
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
        self.Ledger = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox(self)
        self.Lambda = Lambda
        self.Nu = Nu
        self.NodeID = NodeID
        self.Alpha = ALPHA*REP[self.NodeID]/sum(REP)
        self.BackOff = []
        self.LastBackOff = []
        self.LastCongestion = []
        
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
                    self.add_to_inbox(p)
    
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
        if Tran.NodeID == self.NodeID:
            Tran.IssueTime = Time
        # mark this TX as received by this node
        Tran.InformedNodes.append(self)
        if len(Tran.InformedNodes)==NUM_NODES:
            Tran.GlobalSolidTime = Time
        if len(Tran.InformedNodes)==NUM_NODES-1:
            Tran.GlobalSolidTime90pc = Time
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
            if len(self.Inbox.Packets[self.NodeID])>MAX_INBOX_LEN*REP[self.NodeID]+1:
                # ignore it if recently backed off
                if self.LastCongestion:
                    if Time < self.LastCongestion + WAIT_TIME:
                        return
                    
                self.LastCongestion = Time
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
        # token bucket goes here
        
        if Packet.Data not in self.Inbox.Trans:
            if Packet.Data not in self.Ledger:
                self.Inbox.AllPackets.append(Packet)
                self.Inbox.Packets[Packet.Data.NodeID].append(Packet)
                self.Inbox.Trans.append(Packet.Data)
        
    
                
class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self, Node):
        self.Node = Node
        self.AllPackets = []
        self.Packets = []
        self.TokenBucket = []
        self.Tokens = []
        for NodeID in range(NUM_NODES):
            self.Packets.append([])
            self.TokenBucket.append([])
            self.Tokens.append(0)
        self.Trans = []
        
       
    def remove_packet(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.Trans:
            if Packet in self.AllPackets:
                self.AllPackets.remove(Packet)
                self.Packets[Packet.Data.NodeID].remove(Packet)
                self.Trans.remove(Packet.Data)
    
    def has_packets(self, NodeID, Time):
        if self.Packets[NodeID]:
            if self.Packets[NodeID][0].Time<Time or math.isclose(self.Packets[NodeID][0].Time, Time):
                return True
        else:
            return False
        
    def packet_start_times(self, Time1, Time2):
        PacketStartTimes = []
        for NodeID in range(NUM_NODES):
            if self.Packets[NodeID]:
                t = self.Packets[NodeID][0].Time
                if t <= Time1:
                    if Time1 not in PacketStartTimes:
                        PacketStartTimes.append(Time1)
                else:
                    if t not in PacketStartTimes:
                        PacketStartTimes.append(self.Packets[NodeID][0].Time)
        PacketStartTimes.sort()
        if not math.isclose(Time1,Time2):
            PacketStartTimes.append(Time2)
        return PacketStartTimes
    
    def first_finish_time(self, StartTime):
        ActiveNodes = [NodeID for NodeID in range(NUM_NODES) if self.has_packets(NodeID, StartTime)]
        reps = [REP[NodeID] for NodeID in ActiveNodes]
        FirstFinishTime = StartTime+STEP
        for i, NodeID in enumerate(ActiveNodes):
            FinishTime = StartTime + (1/NU-self.Packets[NodeID][0].Proc)*sum(reps)/reps[i]
            if FinishTime<FirstFinishTime:
                FirstFinishTime = FinishTime
        return FirstFinishTime
    
    def process(self, Time):
        if self.AllPackets:
            PacketStartTimes = self.packet_start_times(Time-STEP, Time)
            i = 0
            while len(PacketStartTimes)>1: # all but the last item in the list
                i+=1
                FirstFinishTime = self.first_finish_time(PacketStartTimes[0])
                if FirstFinishTime<PacketStartTimes[1]:
                    ProcTime = FirstFinishTime-PacketStartTimes[0]
                else:
                    ProcTime = PacketStartTimes[1]-PacketStartTimes[0]
                ActiveNodes = [NodeID for NodeID in range(NUM_NODES) if self.has_packets(NodeID, PacketStartTimes[0])]
                reps = [REP[NodeID] for NodeID in ActiveNodes]
                # divide out the processor
                ProcShare = (reps/sum(reps))*ProcTime
                FinishTime = PacketStartTimes[0] + ProcTime
                for j, NodeID in enumerate(ActiveNodes):
                    Packet = self.Packets[NodeID][0]
                    Packet.Proc += ProcShare[j]
                    if math.isclose(Packet.Proc,1/NU):
                        self.remove_packet(Packet)
                        self.Node.add_to_ledger(Packet.TxNode, Packet.Data, FinishTime)
                if math.isclose(FinishTime, Time):
                    break
                else: 
                    PacketStartTimes = self.packet_start_times(FinishTime, Time)
                if i>100:
                    print('infinite loop')
                    break
                        
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
        self.Time = StartTime
        self.Proc = 0
    
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
                t = Packet.Time+self.Delay
                if(t<=Time):
                    self.deliver_packet(Packet, t)
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        """
        When packet has arrived at receiving node, process it
        """
        Packet.Time = Time
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
            Node.Inbox.process(Time)
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
                    SymDiffs[i][j] = len(set(iNode.Ledger).symmetric_difference(set(jNode.Ledger)))
        return SymDiffs + np.transpose(SymDiffs)
    
    def tran_latency(self, latencies):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.IssueTime>200:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.IssueTime)
        return latencies
                
if __name__ == "__main__":
        main()
        
    
    