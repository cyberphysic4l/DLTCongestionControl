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
SIM_TIME = 1200
STEP = 0.1
# Network Parameters
NU = 5
NUM_NODES = 10
NUM_NEIGHBOURS = 4
REP = np.ones(NUM_NODES, dtype=int)
REP[0] = 9
REP[1] = 7
REP[2] = 5
REP[3] = 3
MAX_BURST = 10
AIMD = np.ones(NUM_NODES)
MULT = np.ones(NUM_NODES)
ACTIVE = np.ones(NUM_NODES)
ACTIVE[0] = 0
'''
# dishonest node
MULT[4] = 4
AIMD[4] = 0
'''
# AIMD Parameters
ALPHA = 0.002*NU
BETA = 0.8
WAIT_TIME = 20
MAX_INBOX_LEN = 5 # worst case number of TXs that empty each second

SCHEDULING = 'aimd'
    
def main():
    '''
    Create directory for storing results with these parameters
    '''
    dirstr = 'data/scheduling='+SCHEDULING+'/maxburst='+str(MAX_BURST)+'/nu='+str(NU)+'/rep='+''.join(str(int(e)) for e in REP)+'/active='+''.join(str(int(e)) for e in ACTIVE)+'/aimd='+''.join(str(int(e)) for e in AIMD)+'/multiplier='+''.join(str(int(e)) for e in MULT)+'/alpha='+str(ALPHA)+'/beta='+str(BETA)+'/tau='+str(WAIT_TIME)+'/inbox='+str(MAX_INBOX_LEN)+'/nodes='+str(NUM_NODES)+'/neighbours='+str(NUM_NEIGHBOURS)+'/simtime='+str(SIM_TIME)+'/nmc='+str(MONTE_CARLOS)
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
    Monte Carlo Sims
    """
    Lmds = []
    InboxLens = []
    SymDiffs = []
    latencies = [[] for NodeID in range(NUM_NODES)]
    IATimes = [[] for NodeID in range(NUM_NODES)]
    for mc in range(MONTE_CARLOS):
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
        Net = Network(AdjMatrix)
        # output arrays
        Lmds.append(np.zeros((TimeSteps, NUM_NODES)))
        InboxLens.append(np.zeros((TimeSteps, NUM_NODES)))
        SymDiffs.append(np.zeros(TimeSteps))
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
            for Node in Net.Nodes:
                Lmds[mc][i, Node.NodeID] = sum(Node.Lambda)
                InboxLens[mc][i, Node.NodeID] = len(Net.Nodes[0].Inbox.Packets[Node.NodeID])
            SymDiffs[mc][i] = Net.sym_diffs().max()
        latencies = Net.tran_latency(latencies)
        for NodeID in range(NUM_NODES):
            IATimes[NodeID] += np.diff(Net.Nodes[1].ArrivalTimes[NodeID]).tolist()
            
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
        np.savetxt(dirstr+'/IATimes'+str(NodeID)+'.csv', np.asarray(IATimes[NodeID]), delimiter=',')
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
    IATimes = []
    
    for NodeID in range(NUM_NODES):
        lat = [np.loadtxt(dirstr+'/latencies'+str(NodeID)+'.csv', delimiter=',')]
        if lat:
            latencies.append(lat)
        IAt = [np.loadtxt(dirstr+'/IATimes'+str(NodeID)+'.csv', delimiter=',')]
        if IAt:
            IATimes.append(IAt)
    """
    Plot results
    """
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Rate/Baseline')
    ax1.legend(list(map(str, range(NUM_NODES))))
    for i in range(NUM_NODES):
        ax1.plot(np.arange(0, SIM_TIME, STEP), avgLmds[:,i]*sum(REP)/(NU*REP[i]))
    ax1.plot([0, SIM_TIME], [1, 1], 'r--')
    plt.legend(range(NUM_NODES))
    plt.savefig(dirstr+'/NormalisedLambdas.png')
    
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
        bins = np.arange(0, round(maxval), STEP)
        pdf = np.zeros(len(bins))
        for i, b in enumerate(bins):
            lats = [lat for lat in IATimes[NodeID][0] if (lat>b and lat<b+STEP)]
            pdf[i] = len(lats)
        pdf = pdf/sum(pdf) # normalise
        cdf = np.cumsum(pdf)
        ax6.plot(bins, cdf)
    plt.legend(range(NUM_NODES))
    plt.savefig(dirstr+'/IATimes.png')
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
    def __init__(self, ArrivalTime, Parents, Node, Index=None):
        self.ArrivalTime = ArrivalTime
        self.IssueTime = []
        self.Children = []
        self.Parents = Parents
        self.Index = Index
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
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Ledger = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox()
        self.NodeID = NodeID
        self.Alpha = ALPHA*1/sum(REP)
        self.Lambda = [NU*1/sum(REP) for rep in range(REP[NodeID])]
        self.BackOff = [[] for rep in range(REP[NodeID])]
        self.LastBackOff = [[] for rep in range(REP[NodeID])]
        self.ArrivalTimes = [[] for NodeID in range(NUM_NODES)]
        
    def generate_txs(self, Time, Index):
        """
        Create new TXs at rate lambda and do PoW
        """
        if ACTIVE[self.NodeID]:
            NewTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda[Index])))
        else:
            NewTXs = []
            self.Lambda[Index] = 0
        for t in NewTXs:
            Parents = self.select_tips()
            self.TempTransactions.append(Transaction(t, Parents, self, Index))
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
            Selection = self.Ledger[-2:-1]
        return Selection
    
    def process_inbox(self, Time):
        """
        Process TXs in inbox of TXs received from neighbours
        """
        # sort inboxes by arrival time
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        nTX = np.random.poisson(STEP*NU)
        times = np.sort(np.random.uniform(Time, Time+STEP, nTX))
        i = 0
        while i < nTX:
            if SCHEDULING=='fifo':
                Packet = self.Inbox.fifo_schedule(Time)
            elif SCHEDULING=='aimd':
                Packet = self.Inbox.aimd_schedule(Time)
            elif SCHEDULING=='tokenbucket':
                Packet = self.Inbox.token_bucket_schedule(Time)
            elif SCHEDULING=='wrr':
                Packet = self.Inbox.wrr_schedule(Time)
            elif SCHEDULING=='brr':
                Packet = self.Inbox.brr_schedule(Time)
            elif SCHEDULING=='drr':
                Packet = self.Inbox.drr_schedule()
            elif SCHEDULING=='bob':
                Packet = self.Inbox.bob_schedule(Time)
            if Packet is not None:
                if Packet.Data not in self.Ledger:
                    self.add_to_ledger(Packet.TxNode, Packet.Data, times[i])
                i += 1
            else:
                break
    
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
            for i in range(REP[self.NodeID]):
                Packets = [p for p in self.Inbox.Packets[self.NodeID] if p.Data.Index==i]
                if len(Packets)>MAX_INBOX_LEN:
                    self.BackOff[i] = True
            
    def set_rate(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        for i in range(REP[self.NodeID]):
            if AIMD[self.NodeID] and Time>=60: # AIMD starts after 1 min adjustment
                # if wait time has not passed---reset.
                if self.LastBackOff[i]:
                    if Time < self.LastBackOff[i] + WAIT_TIME:
                        self.BackOff[i] = False
                        continue
                # multiplicative decrease or else additive increase
                if self.BackOff[i]:
                    self.Lambda[i] = self.Lambda[i]*BETA
                    self.LastBackOff[i] = Time
                else:
                    self.Lambda[i] += MULT[self.NodeID]*self.Alpha*STEP
            else:
                self.Lambda[i] = MULT[self.NodeID]*NU*1/sum(REP)
            
    def add_to_inbox(self, Packet):
        """
        Add to inbox if not already received and/or processed
        """
        if Packet.Data not in self.Inbox.Trans:
            if Packet.Data not in self.Ledger:
                self.Inbox.AllPackets.append(Packet)
                self.Inbox.Packets[Packet.Data.NodeID].append(Packet)
                self.Inbox.Trans.append(Packet.Data)
                self.ArrivalTimes[Packet.Data.NodeID].append(Packet.EndTime)
        
    
                
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
        self.RRSlot = 0
        self.RRNodeID = 0
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
        population = [NodeID for NodeID in range(NUM_NODES) if ACTIVE[NodeID]]
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
            
        
    def drr_schedule(self):
        if self.AllPackets:
            while True:
                if self.RRSlot >= REP[self.RRNodeID]:
                    self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
                    self.RRSlot = 0
                if self.Packets[self.RRNodeID]:
                    Packet = self.Packets[self.RRNodeID][0]
                    if self.Deficit[self.RRNodeID]:
                        self.Deficit[self.RRNodeID] -= 1
                    else:
                        self.RRSlot += 1
                    # remove the transaction from all inboxes
                    self.remove_packet(Packet)
                    return Packet
                else: # move to next node's first slot
                    self.RRSlot += 1
                    self.Deficit[self.RRNodeID] = min(self.Deficit[self.RRNodeID]+1, MAX_BURST*REP[self.RRNodeID]) # limited deficit savings

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
    def __init__(self, AdjMatrix):
        self.A = AdjMatrix
        self.Nodes = []
        self.CommChannels = []
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
            for i in range(REP[Node.NodeID]):
                Node.generate_txs(Time, i)
            Node.process_inbox(Time)
            Node.check_congestion(Time)
            Node.set_rate(Time)
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
    
    def tran_latency(self, latencies):
        for Tran in self.Nodes[0].Ledger:
            if Tran.GlobalSolidTime and Tran.IssueTime>200:
                latencies[Tran.NodeID].append(Tran.GlobalSolidTime-Tran.IssueTime)
        return latencies
                
if __name__ == "__main__":
        main()
        
    
    