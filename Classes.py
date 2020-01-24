# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019

@author: Pietro Ferraro and Andrew Cullen
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

SIM_TIME = 600
STEP = 0.1
WAIT_TIME = 5
MAX_INBOX_LEN = 20
NUM_NODES = 8
NUM_NEIGHBOURS = 4
# global params
MANA = np.ones(NUM_NODES) 
ALPHA = 1
BETA = 0.7
NU = 10
    
def main():
    """
    Setup simulation inputs and instantiate output arrays
    """
    TimeSteps = int(SIM_TIME/STEP)
    # Generate network topology
    G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES)
    # Get adjacency matrix and weight by delay at each channel
    ChannelDelays = 0.9*np.ones((NUM_NODES, NUM_NODES))+0.2*np.random.rand(NUM_NODES, NUM_NODES)
    AdjMatrix = np.multiply(1*np.asarray(nx.to_numpy_matrix(G)), ChannelDelays)
    # Node parameters
    Lambdas = 0.1*(np.ones(NUM_NODES))
    Nus = NU*(np.ones(NUM_NODES))
    
    # Initialise output arrays
    Tips = np.zeros((TimeSteps, NUM_NODES))
    QLen = np.zeros((TimeSteps, NUM_NODES))
    Lmds = [np.zeros((TimeSteps, NUM_NODES))]
    for i in range(NUM_NEIGHBOURS):
        Lmds.append(np.zeros((TimeSteps, NUM_NODES)))
    OBs = [np.zeros((TimeSteps, NUM_NODES))]
    for i in range(NUM_NEIGHBOURS-1):
        OBs.append(np.zeros((TimeSteps, NUM_NODES)))
    AvgSymDiffs = np.zeros((TimeSteps, NUM_NODES))
    """
    Run simulation for the specified time
    """
    Net = Network(AdjMatrix, Lambdas, Nus)
    for i in range(TimeSteps):
        # discrete time step size specified by global variable STEP
        T = STEP*i
        if (T>500):
            for Node in Net.Nodes:
                Node.Lambdas[0] = 0
        # update network for all new events in this time step
        Net.simulate(T)
        # save summary results in output arrays
        for Node in Net.Nodes:
            Tips[i, Node.NodeID] = len(Node.TipsSet)
            QLen[i, Node.NodeID] = len(Node.Inbox)
            Lmds[0][i, Node.NodeID] = Node.Lambdas[0]
            for j, Neighbour in enumerate(Node.Neighbours):
                Lmds[j+1][i, Node.NodeID] = Node.Lambdas[j+1]
            for j, Neighbour in enumerate(Node.Neighbours):
                OBs[j][i, Node.NodeID] = len(Node.Outboxes[j])
        AvgSymDiffs[i,:] = np.average(Net.sym_diffs(), axis=0)
    print(np.average(Lmds[0], axis=0))
    
    """
    Plot results
    """
    plt.close('all')
    fig, ax1 = plt.subplots(4, 1)
    
    for i, Node in enumerate(Net.Nodes):
        ax1[0].plot(np.arange(0, TimeSteps*STEP, STEP), AvgSymDiffs[:,Node.NodeID], color=colors[i])
    ax1[0].set_xlabel('Time')
    ax1[0].set_ylabel('Avg Symmetric Diff')
    ax1[0].legend(list(map(str, range(NUM_NODES))))
    
    for Node in Net.Nodes:
        ax1[1].plot(np.arange(0, TimeSteps*STEP, STEP), Tips[:,Node.NodeID])
    ax1[1].set_xlabel('Time')
    ax1[1].set_ylabel('Number of Tips')
    ax1[1].legend(list(map(str, range(NUM_NODES))))
    
    for Node in Net.Nodes:
        ax1[2].plot(np.arange(0, TimeSteps*STEP, STEP), QLen[:,Node.NodeID])
    ax1[2].set_xlabel('Time')
    ax1[2].set_ylabel('Inbox Length')
    ax1[2].legend(list(map(str, range(NUM_NODES))))
    
    for Node in Net.Nodes:
        ax1[3].plot(np.arange(0, TimeSteps*STEP, STEP), Lmds[0][:,Node.NodeID])
    ax1[3].set_xlabel('Time')
    ax1[3].set_ylabel('Lambda')
    ax1[3].legend(list(map(str, range(NUM_NODES))))
    
    fig2, ax2 = plt.subplots(NUM_NODES, 1)
    for i, Node in enumerate(Net.Nodes):
        for j, Neighbour in enumerate(Node.Neighbours):
            ax2[i].plot(np.arange(0, TimeSteps*STEP, STEP), Lmds[j+1][:,Node.NodeID], color=colors[Neighbour.NodeID])
    
    fig3, ax3 = plt.subplots(NUM_NODES, 1)
    for i, Node in enumerate(Net.Nodes):
        for j, Neighbour in enumerate(Node.Neighbours):
            ax3[i].plot(np.arange(0, TimeSteps*STEP, STEP), OBs[j][:,Node.NodeID], color=colors[Neighbour.NodeID])
    
    plt.show()
    plt.figure()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_color=colors[0:NUM_NODES])
    plt.show()


class Transaction:
    """
    Object to simulate a transaction its edges in the DAG
    """
    def __init__(self, ArrivalTime, Parents, NodeID):
        self.ArrivalTime = ArrivalTime
        self.Children = []
        self.Parents = Parents
        self.NodeID = NodeID # signature of issuing node
        
    def is_tip(self):
        if not self.Children:
            return True
        else:
            return False

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, Lambdas, Nu, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Tangle = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Neighbours = []
        self.Network = Network
        self.Inbox = []
        self.Outboxes = []
        self.Lambdas = [Lambdas]
        self.Nu = Nu
        self.NodeID = NodeID
        self.BackOff = [[]]
        self.LastBackOff = [[]]
        self.LastCongestion = []
        
    def generate_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        NewTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambdas[0])))
        for t in NewTXs:
            Parents = self.select_tips(2)
            self.TempTransactions.append(Transaction(t, Parents, self.NodeID))
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
    
    def select_tips(self, NumberOfSelections):
        """
        Implements uniform random tip selection
        """
        Selection = []
        NumberOfTips= len(self.TipsSet)
        for i in range(NumberOfSelections):
            Selection.append(self.TipsSet[np.random.randint(NumberOfTips)])
        return Selection
    
    def process_inbox(self, Time):
        """
        Process TXs in inbox of TXs received from neighbours
        """
        # first, sort the inbox by the time the TXs arrived there (FIFO)
        self.Inbox.sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        nTX = np.random.poisson(STEP*self.Nu)
        i = 0
        while i < nTX:
            if self.Inbox:
                if self.Inbox[0].Data not in self.Tangle:
                    self.Tangle.append(self.Inbox[0].Data)
                    if not self.Inbox[0].Data.Children:
                        self.TipsSet.append(self.Inbox[0].Data)
                    if self.Inbox[0].Data.Parents:
                        for Parent in self.Inbox[0].Data.Parents:
                            Parent.Children.append(self.Inbox[0].Data)
                            if Parent in self.TipsSet:
                                self.TipsSet.remove(Parent)
                            else:
                                continue
                    else:
                        pass
                    # add these processed TXs to outbox
                    for Outbox in self.Outboxes:
                        Outbox.append(self.Inbox[0].Data)
                    del self.Inbox[0]
                    i += 1
                else:
                    del self.Inbox[0]
            else:
                break
                
            
    def process_outboxes(self, Time):
        """
        Process TXs in outbox corresponding to each neighbour
        """
        for i, Outbox in enumerate(self.Outboxes):
            nTX = np.random.poisson(STEP*self.Lambdas[i+1])
            ProcessedTXs = np.sort(np.random.uniform(Time, Time+STEP, nTX))
            for t in ProcessedTXs:
                if Outbox:
                    self.Network.send_data(self, self.Neighbours[i], Outbox[0], t)
                    del Outbox[0]
                else:
                    break
    
    def check_congestion(self, Time):
        """
        Check if congestion is occuring and send back-offs
        """
        if len(self.Inbox) > MAX_INBOX_LEN:
            if self.LastCongestion:
                if Time < self.LastCongestion + WAIT_TIME:
                    return
            # count number of TXs from each neighbour (and self) in the inbox
            NodeTrans = np.zeros(len(self.Neighbours)+1)
            for Packet in self.Inbox:
                IssuingNodeID = Packet.Data.NodeID
                if Packet.TxNode == self:
                    NodeTrans[0] += 100/MANA[IssuingNodeID]
                else:
                    index = self.Neighbours.index(Packet.TxNode)+1
                    NodeTrans[index] += 1/MANA[IssuingNodeID]
            # Probability of backing off
            Probs = (NodeTrans)/sum(NodeTrans)
            randIndex = np.random.choice(range(len(Probs)), p=Probs)
            if randIndex == 0: # this node itself must back off
                self.BackOff[0] = True
            else:
                RxNode = self.Neighbours[randIndex-1]
                self.Network.send_data(self, RxNode, 'back off', Time)
            
    def process_cong_notif(self, Packet, Time):
        """
        Process a received congestion notification
        """
        index = self.Neighbours.index(Packet.TxNode)+1
        self.BackOff[index] = True
            
    def aimd_update(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        for i in range(len(self.Neighbours)+1):
            if self.LastBackOff[i]:
                if Time < self.LastBackOff[i] + WAIT_TIME:
                    self.BackOff[i] = False
                    continue
            if i==0: # for self
                Alpha = ALPHA*MANA[self.NodeID]/sum(MANA)
            else: # for neighbours
                Alpha = ALPHA
            if self.BackOff[i]:
                self.Lambdas[i] = self.Lambdas[i]*BETA
                self.LastBackOff[i] = Time
            else:
                self.Lambdas[i] += Alpha*STEP
            
                
    def add_to_inbox(self, Packet):
        """
        Add to inbox if not already received and/or processed
        """
        self.Inbox.append(Packet)
        

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
                self.Nodes[i].Outboxes.append([]) # start with empty outbox for each neighbour
                self.Nodes[i].Lambdas.append(NUM_NODES*Lambdas[i]) # start with lambda for each outbox
                self.Nodes[i].BackOff.append([])
                self.Nodes[i].LastBackOff.append([])
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.A[i][j]))
            self.CommChannels.append(RowList)
            
    def send_data(self, TxNode, RxNode, Data, Time):
        """
        Send this data (TX or back off) to a specified neighbour
        """
        CC = self.CommChannels[TxNode.NodeID][TxNode.Neighbours.index(RxNode)]
        CC.send_packet(TxNode, RxNode, Data, Time)
            
    def broadcast_data(self, TxNode, Data, Time):
        """
        Send this data (TX or back off) to all neighbours
        """
        for i, CC in enumerate(self.CommChannels[self.Nodes.index(TxNode)]):
            CC.send_packet(TxNode, TxNode.Neighbours[i], Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate and process new transactions
        """
        for Node in self.Nodes:
            Node.generate_txs(Time)
            Node.process_inbox(Time)
            Node.process_outboxes(Time)
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
            
                
if __name__ == "__main__":
        main()
        
    
    