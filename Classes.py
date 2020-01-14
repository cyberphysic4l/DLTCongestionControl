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
    
def main():
    """
    Setup simulation inputs and instantiate output arrays
    """
    TimeSteps = SIM_TIME/STEP
    # Generate network topology
    G = nx.random_regular_graph(NUM_NEIGHBOURS, NUM_NODES)
    # Get adjacency matrix and weight by delay at each channel
    AdjMatrix = 1*np.asarray(nx.to_numpy_matrix(G))
    # Node parameters
    Lambdas = 0.1*(np.ones(NUM_NODES))
    Nus = 20*(np.ones(NUM_NODES))
    Alphas = 0.1*(np.ones(NUM_NODES))
    Betas = 0.7*(np.ones(NUM_NODES))
    Manas = np.ones(NUM_NODES)
    # Initialise output arrays
    Tips = np.zeros((TimeSteps, NUM_NODES))
    QLen = np.zeros((TimeSteps, NUM_NODES))
    Lmds = np.zeros((TimeSteps, NUM_NODES))
    
    """
    Run simulation for the specified time
    """
    Net = Network(AdjMatrix, Lambdas, Nus, Alphas, Betas, Manas)
    for i in range(TimeSteps):
        # discrete time step size specified by global variable STEP
        T = STEP*i
        # next two lines generate new transactions for this time step and update network for all new events
        Net.generate_transactions(T)
        Net.update_comm_channels(T)
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

class BackOff:
    """
    *Not used yet*
    This class represents the data that would need to be sent in a
    packet to instruct a specific node to back off its send rate
    """
    def _init_(self, RXNodeID):
        self.RXNodeID = RXNodeID
        
    def get_rxnode_id(self):
        return self.RXNodeID

class Node:
    
    def __init__(self, Network, Lambda, Nu, Alpha, Beta, Mana, NodeID, Genesis, PoWDelay = 1):
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
        self.LastBackOff = []
        self.LastCongestion = []
        self.Mana = Mana
        
    def report(self):
        return None
        
        
    def select_tips(self, NumberOfSelections):
        """
        Implements uniform random tip selection
        """
        Selection = []
        NumberOfTips= len(self.TipsSet)
        for i in range(NumberOfSelections):
            Selection.append(self.TipsSet[np.random.randint(NumberOfTips)])
        return Selection

    def add_transactions(self, Time):
        """
        This method includes:
            1. PoW and broadcast of newly added TXs
            2. Processing queue of transactions received from neighbours
            3. Checking queue length for congestion events and send back-offs
        """
        # 1. process newly created TXs when finished PoW
        for Tran in self.TempTransactions:
            if (Tran.ArrivalTime + self.PoWDelay) <= Time:
                self.TempTransactions.remove(Tran)
                self.Tangle.append(Tran)
                self.TipsSet.append(Tran)
                for Parent in Tran.Parents:
                    Parent.Children.append(Tran)
                    if Parent in self.TipsSet:
                        self.TipsSet.remove(Parent)
                    else:
                        continue
                self.Network.broadcast_data(self, Tran, Time)
        
        # 2. process TXs in queue of TXs received from neighbours
        for i in range(np.random.poisson(STEP*self.Nu)):
            if i >= len(self.Queue):
                break
            self.Tangle.append(self.Queue[i])
            if not self.Queue[i].Children:
                self.TipsSet.append(self.Queue[i])
            if self.Queue[i].Parents:
                for Parent in self.Queue[i].Parents:
                    Parent.Children.append(self.Queue[i])
                    if Parent in self.TipsSet:
                        self.TipsSet.remove(Parent)
                    else:
                        continue
            else:
                pass
            self.Network.broadcast_data(self, self.Queue[i], Time)
            del self.Queue[i]
            
        # 3. check if congestion is occuring and send back-offs
        if len(self.Queue) > MAX_QUEUE_LEN:
            if self.LastCongestion:
                if self.LastCongestion < Time-WAIT_TIME:
                    NodeTrans = np.zeros(NUM_NODES)
                    for Tran in self.Queue:
                        NodeTrans[Tran.NodeID] += 1 
            self.Network.broadcast_data(self, 'back off', Time)
                
    def add_to_queue(self, Tran):
        """
        Includes basic prefilter to remove transactions already processed
        """
        if (Tran not in self.Queue) and (Tran not in self.Tangle):
            self.Queue.append(Tran)
            
    def increase_lambda(self):
        """
        AIMD additive increase
        """
        self.Lambda += self.Alpha*STEP
        
    def back_off(self, Time):
        """
        AIMD multiplicative decrease
        Includes check of wait time etc
        """
        if self.LastBackOff:
            if self.LastBackOff < Time-WAIT_TIME:
                self.Lambda = self.Lambda*self.Beta
                self.LastBackOff = Time
        else:
            self.Lambda = self.Lambda*self.Beta
            self.LastBackOff = Time
    
        
class Packet:
    """
    Object for sending data including TXs and back off notifications over
    comm channels
    """    
    def __init__(self, Data, StartTime):
        # can be a TX or a back off notification
        self.Data = Data
        self.StartTime = StartTime
        
    def get_start_time(self):
        return self.StartTime
    
    
    def get_data(self):
        return self.Data
    

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
                if(Time>=Packet.get_start_time()+self.Delay):
                    self.deliver_packet(Packet, Time)
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        """
        When packet has arrived at receiving node, process it
        """
        Data = Packet.get_data()
        if isinstance(Data, Transaction):
            # if this is a transaction, add it to queue
            self.RxNode.add_to_queue(Data)
        else: 
            # else this is a back off notification
            self.RxNode.back_off(Time)
        self.Packets.remove(Packet)
        
class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix, Lambdas, Nus, Alphas, Betas, Manas):
        self.A = AdjMatrix
        self.Lambdas = Lambdas
        self.Nodes = []
        self.CommChannels = []
        Genesis = Transaction(0, [], [])
        
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(Node(self, Lambdas[i], Nus[i], Alphas[i], Betas[i], Manas[i], i, Genesis))
           
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
        
    def update_comm_channels(self, Time):
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time)
    
    def generate_transactions(self, Time):
        """
        Each node generate and process new transactions
        """
        for Node in self.Nodes:
            for i in range(np.random.poisson(STEP*Node.Lambda)):
                Parents = Node.select_tips(2)
                Node.TempTransactions.append(Transaction(Time, Parents, Node.NodeID))
            Node.add_transactions(Time)
            Node.increase_lambda() # AIMD update
            

if __name__ == "__main__":
        main()
        
    
    