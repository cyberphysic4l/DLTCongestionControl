# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019

@author: Pietro Ferraro and Andrew Cullen
"""
import numpy as np
import matplotlib.pyplot as plt

STEP = 0.1
WAIT_TIME = 5
    
def main():
    NumNodes = 4
    TimeSteps = 6000
    DelayMatrix = (np.ones((NumNodes, NumNodes)) - np.identity(NumNodes)) # all-to-all
    Lambdas = 0.01*(np.ones(NumNodes))
    Nus = 10*(np.ones(NumNodes))
    Alphas = 0.01*(np.ones(NumNodes))
    Betas = 0.7*(np.ones(NumNodes))
    Net = Network(DelayMatrix, Lambdas, Nus, Alphas, Betas)
    Tips = np.zeros((TimeSteps, NumNodes))
    QLen = np.zeros((TimeSteps, NumNodes))
    Lmds = np.zeros((TimeSteps, NumNodes))
    
    for i in range(TimeSteps):
        T = STEP*i
        Net.generate_transactions(T)
        Net.update_comm_channels(T)
        for Node in Net.Nodes:
            Tips[i, Node.NodeID] = len(Node.TipsSet)
            QLen[i, Node.NodeID] = len(Node.Queue)
            Lmds[i, Node.NodeID] = Node.Lambda
    
    plt.close('all')
    plt.figure(1)
    for Node in Net.Nodes:
        plt.plot(np.arange(0, TimeSteps*STEP, STEP), Tips[:,Node.NodeID])
    plt.xlabel('Time')
    plt.ylabel('Number of Tips')
    plt.show()
    
    plt.figure(2)
    for Node in Net.Nodes:
        plt.plot(np.arange(0, TimeSteps*STEP, STEP), QLen[:,Node.NodeID])
    plt.xlabel('Time')
    plt.ylabel('Queue Length')
    plt.show()
    
    plt.figure(3)
    for Node in Net.Nodes:
        plt.plot(np.arange(0, TimeSteps*STEP, STEP), Lmds[:,Node.NodeID])
    plt.xlabel('Time')
    plt.ylabel('Lambda')
    plt.show()


class Transaction:
    
    def __init__(self, ArrivalTime, Parents, Signature):
        self.ArrivalTime = ArrivalTime
        self.Children = []
        self.Parents = Parents
        self.NodeSignature = Signature
        
    def is_tip(self):
        if not self.Children:
            return True
        else:
            return False
        

class Node:
    
    def __init__(self, Network, Lambda, Nu, Alpha, Beta, NodeID, Genesis, PoWDelay = 1, MaxQueueLen = 20):
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
        self.MaxQueueLen = MaxQueueLen
        self.LastBackOff = []
        
    def report(self):
        return None
        
        
    def select_tips(self, NumberOfSelections):
        Selection = []
        NumberOfTips= len(self.TipsSet)
        # uniform random tip selection
        for i in range(NumberOfSelections):
            Selection.append(self.TipsSet[np.random.randint(NumberOfTips)])
        return Selection

    def add_transactions(self, Time):
        
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
            
        if len(self.Queue) > self.MaxQueueLen:
            self.Network.broadcast_data(self, 'back off', Time)
                
    def add_to_queue(self, Tran):
        if (Tran not in self.Queue) and (Tran not in self.Tangle):
            self.Queue.append(Tran)
            
    def increase_lambda(self):
        self.Lambda += self.Alpha
        
    def back_off(self, Time):
        if self.LastBackOff:
            if self.LastBackOff < Time-WAIT_TIME:
                self.Lambda = self.Lambda*self.Beta
                self.LastBackOff = Time
        else:
            self.Lambda = self.Lambda*self.Beta
            self.LastBackOff = Time
    
        
class Packet:
    
    def __init__(self, Data, StartTime):
        self.Data = Data
        self.StartTime = StartTime
        
    def get_start_time(self):
        return self.StartTime
    
    
    def get_data(self):
        return self.Data
    

class CommChannel:
    
    def __init__(self, TxNode, RxNode, Delay):
        self.TxNode = TxNode
        self.RxNode = RxNode
        self.Delay = Delay
        self.Packets = []
    
    def send_packet(self, Data, Time):
        self.Packets.append(Packet(Data, Time))
    
    def transmit_packets(self, Time):
        if self.Packets:
            for Packet in self.Packets:
                if(Time>=Packet.get_start_time()+self.Delay):
                    self.deliver_packet(Packet, Time)
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        Data = Packet.get_data()
        if isinstance(Data, Transaction): # if this is a transaction, add it to queue
            self.RxNode.add_to_queue(Data)
        else: # else this is a back off notification
            self.RxNode.back_off(Time)
        self.Packets.remove(Packet)
        
class Network:
    
    def __init__(self, DelayMatrix, Lambdas, Nus, Alphas, Betas):
        self.D = DelayMatrix
        self.Lambdas = Lambdas
        self.Nodes = []
        self.CommChannels = []
        Genesis = Transaction(0, [], [])
        
        for i in range(np.size(self.D,1)):
            self.Nodes.append(Node(self, Lambdas[i], Nus[i], Alphas[i], Betas[i], i, Genesis))
            
        for i in range(np.size(self.D,1)):
            RowList = []
            for j in np.nditer(np.nonzero(self.D[i,:])):
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.D[i][j]))
            self.CommChannels.append(RowList)
            
    def broadcast_data(self, Node, Data, Time):
        for CommChannel in self.CommChannels[self.Nodes.index(Node)]:
            CommChannel.send_packet(Data, Time)
        
    def update_comm_channels(self, Time):
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time)
    
    def generate_transactions(self, Time):
        for Node in self.Nodes:
            for i in range(np.random.poisson(STEP*Node.Lambda)):
                Parents = Node.select_tips(2)
                Node.TempTransactions.append(Transaction(Time, Parents, Node.NodeID))
            Node.add_transactions(Time)
            Node.increase_lambda() # AIMD update
            

if __name__ == "__main__":
        main()
        
    
    