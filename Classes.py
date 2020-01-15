# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019

@author: Pietro Ferraro and Andrew Cullen
"""
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

SIM_TIME = 600
STEP = 0.1
WAIT_TIME = 5
MAX_QUEUE_LEN = 20
NUM_NODES = 10
NUM_NEIGHBOURS = 4
MANA = np.ones(NUM_NODES) # Assume Mana is global
MANA[0:5] = 10*np.ones(5)
    
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
    Nus = 20*(np.ones(NUM_NODES))
    Alphas = 0.01*(np.ones(NUM_NODES))
    Betas = 0.7*(np.ones(NUM_NODES))
    
    # Initialise output arrays
    Tips = np.zeros((TimeSteps, NUM_NODES))
    QLen = np.zeros((TimeSteps, NUM_NODES))
    Lmds = np.zeros((TimeSteps, NUM_NODES))
    
    """
    Run simulation for the specified time
    """
    Net = Network(AdjMatrix, Lambdas, Nus, Alphas, Betas)
    for i in range(TimeSteps):
        # discrete time step size specified by global variable STEP
        T = STEP*i
        # update network for all new events in this time step
        Net.simulate(T)
        # save summary results in output arrays
        for Node in Net.Nodes:
            Tips[i, Node.NodeID] = len(Node.TipsSet)
            QLen[i, Node.NodeID] = len(Node.Queue)
            Lmds[i, Node.NodeID] = Node.Lambda
    
    """
    Plot results
    """
    plt.close('all')
    fig, ax = plt.subplots(3, 1)
    for Node in Net.Nodes:
        ax[0].plot(np.arange(0, TimeSteps*STEP, STEP), Tips[:,Node.NodeID])
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Number of Tips')
    
    for Node in Net.Nodes:
        ax[1].plot(np.arange(0, TimeSteps*STEP, STEP), QLen[:,Node.NodeID])
    ax[1].set_xlabel('Time')
    ax[1].set_ylabel('Queue Length')
    
    for Node in Net.Nodes:
        ax[2].plot(np.arange(0, TimeSteps*STEP, STEP), Lmds[:,Node.NodeID])
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Lambda')
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
    def __init__(self, Network, Lambda, Nu, Alpha, Beta, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Tangle = [Genesis]
        self.TempTransactions = []
        self.PoWDelay = PoWDelay
        self.Users = []
        self.Network = Network
        self.Queue = []
        self.Lambda = Lambda
        self.Nu = Nu
        self.Alpha = Alpha
        self.Beta = Beta
        self.NodeID = NodeID
        self.BackOff = False
        self.LastBackOff = []
        self.LastCongestion = []
        self.CongNotifs = []
        
    def select_tips(self, NumberOfSelections):
        """
        Implements uniform random tip selection
        """
        Selection = []
        NumberOfTips= len(self.TipsSet)
        for i in range(NumberOfSelections):
            Selection.append(self.TipsSet[np.random.randint(NumberOfTips)])
        return Selection

    def process_own_txs(self, Time):
        """
        Process pending TXs as they finish PoW
        """
        if self.TempTransactions:
            for Tran in self.TempTransactions:
                t = Tran.ArrivalTime + self.PoWDelay
                if t <= Time:
                    self.TempTransactions.remove(Tran)
                    self.Tangle.append(Tran)
                    self.TipsSet.append(Tran)
                    for Parent in Tran.Parents:
                        Parent.Children.append(Tran)
                        if Parent in self.TipsSet:
                            self.TipsSet.remove(Parent)
                        else:
                            continue
                    self.Network.broadcast_data(self, Tran, t)
    
    def process_queue(self, Time):
        """
        Process TXs in queue of TXs received from neighbours
        """
        # first, sort the queue by the time the TXs arrived there (FIFO)
        self.Queue.sort(key=lambda p: p.EndTime)
        ProcessedTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Nu)))
        for i, t in enumerate(ProcessedTXs):
            if i >= len(self.Queue):
                break
            self.Tangle.append(self.Queue[i].Data)
            if not self.Queue[i].Data.Children:
                self.TipsSet.append(self.Queue[i].Data)
            if self.Queue[i].Data.Parents:
                for Parent in self.Queue[i].Data.Parents:
                    Parent.Children.append(self.Queue[i].Data)
                    if Parent in self.TipsSet:
                        self.TipsSet.remove(Parent)
                    else:
                        continue
            else:
                pass
            self.Network.broadcast_data(self, self.Queue[i].Data, t)
            del self.Queue[i]
    
    def check_congestion(self, Time):
        """
        Check if congestion is occuring and send back-offs
        """
        if len(self.Queue) > MAX_QUEUE_LEN:
            if self.LastCongestion:
                if Time < self.LastCongestion + WAIT_TIME:
                    return
            # count number of TXs from each node in the queue
            NodeTrans = np.zeros(NUM_NODES)
            for Packet in self.Queue:
                NodeTrans[Packet.Data.NodeID] += 1
            
            # Probability of backing off
            Probs = (NodeTrans/MANA)/sum(NodeTrans/MANA)
            RxNodeID = np.random.choice(range(NUM_NODES), p=Probs)
            CongNotif = CongestionNotification(self.NodeID, RxNodeID, Time)
            self.CongNotifs.append(CongNotif)
            self.Network.broadcast_data(self, CongNotif, Time)
            
    def process_cong_notif(self, Packet, Time):
        """
        Process a received congestion notification
        """
        CongNotif = Packet.Data
        if CongNotif not in self.CongNotifs:
            self.CongNotifs.append(CongNotif)
            if CongNotif.RxNodeID == self.NodeID:
                self.BackOff = True
            else:
                self.Network.broadcast_data(self, CongNotif, Time)
            
    def aimd_update(self, Time):
        """
        Additively increase of multiplicatively decrease lambda
        """
        if self.LastBackOff:
            if Time < self.LastBackOff + WAIT_TIME:
                self.BackOff = False
                return
        if self.BackOff:
            self.Lambda = self.Lambda*self.Beta
            self.LastBackOff = Time
        else:
            self.Lambda += self.Alpha*STEP
            
                
    def add_to_queue(self, Packet):
        """
        Add to queue if not already received and/or processed
        """
        Tran = Packet.Data
        # check TX is not already in the queue in different packet
        for p in self.Queue:
            if p.Data == Tran:
                if Packet.EndTime < p.EndTime:
                    self.Queue.remove(p)
                    self.Queue.append(Packet)
                return
        # check TX is not already in this node's Tangle
        if Tran not in self.Tangle:
            self.Queue.append(Packet)
        
class CongestionNotification:
    """
    Object for data to be sent to tell nodes to back off etc
    """
    def __init__(self, TxNodeID, RxNodeID, CongTime):
        self.TxNodeID = TxNodeID
        self.RxNodeID = RxNodeID
        self.CongTime = CongTime

class Packet:
    """
    Object for sending data including TXs and back off notifications over
    comm channels
    """    
    def __init__(self, Data, StartTime):
        # can be a TX or a back off notification
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
    
    def send_packet(self, Data, Time):
        """
        Add new packet to the comm channel with time of arrival
        """
        self.Packets.append(Packet(Data, Time))
    
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
            # if this is a transaction, add the Packet to queue
            self.RxNode.add_to_queue(Packet)
        elif isinstance(Packet.Data, CongestionNotification): 
            # else this is a back off notification
            self.RxNode.process_cong_notif(Packet, Time)
        self.Packets.remove(Packet)
        
class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix, Lambdas, Nus, Alphas, Betas):
        self.A = AdjMatrix
        self.Lambdas = Lambdas
        self.Nodes = []
        self.CommChannels = []
        Genesis = Transaction(0, [], [])
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(Node(self, Lambdas[i], Nus[i], Alphas[i], Betas[i], i, Genesis))
        # Create list of comm channels corresponding to each node
        for i in range(np.size(self.A,1)):
            RowList = []
            for j in np.nditer(np.nonzero(self.A[i,:])):
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.A[i][j]))
            self.CommChannels.append(RowList)
            
    def broadcast_data(self, Node, Data, Time):
        """
        Send this data (TX or back off) to all neighbours
        """
        for CommChannel in self.CommChannels[self.Nodes.index(Node)]:
            CommChannel.send_packet(Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate and process new transactions
        """
        for Node in self.Nodes:
            NewTXs = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*Node.Lambda)))
            for t in NewTXs:
                Parents = Node.select_tips(2)
                Node.TempTransactions.append(Transaction(t, Parents, Node.NodeID))
            Node.process_own_txs(Time)
            Node.process_queue(Time)
            Node.check_congestion(Time)
            Node.aimd_update(Time)
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time)

if __name__ == "__main__":
        main()
        
    
    