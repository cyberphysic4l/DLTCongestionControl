# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019

@author: Pietro Ferraro
"""
import numpy as np

def main():
    NumNodes = 3
    DelayMatrix = (np.ones(NumNodes, NumNodes) - np.identity(NumNodes)) # all-to-all
    Lambdas = 5*(np.ones(NumNodes))
    Network = Network(DelayMatrix)
    
    # some loop over time
    NodeLambda = 10 # each Node receiving TXs to add at this rate
    Network.process_transactions(NodeLambda, Time)
    

class Transaction:
    
    def __init__(self, ArrivalTime, Parents, No = None, Signature):
        self.No = No
        self.ArrivalTime = ArrivalTime
        self.Children = []
        self.Parents = Parents
        self.NodeSignature = Signature
        
    def is_tip(self):
        if not Children:
            return True
        else:
            return False
        

class Node:
    
    def __init__(self, Network, Lambda, PoWDelay = 1):
        self.TipsSet = []
        self.TempTransactions = []
        self.Tangle = []
        self.PoWDelay = PoWDelay
        self.Users = []
        self.Network = Network
        self.Queue = []
        self.Lambda = Lambda
        
        self.TipsSet.append(Transaction(0, [], 0))
        self.Tangle.append(Transaction(0, [], 0)) # add genesis
        
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
                Tran.No = Time + np.random.normal(0,0.01)
                self.TempTransactions.remove(Tran)
                self.Tangle.append(Tran)
                self.TipsSet.append(Tran)
                for Parent in Tran.Parents:
                    Parent.Children.append(Tran)
                    if Parent in self.TipsSet:
                        self.TipsSet.remove(Child)
                    else:
                        continue
                self.Network.broadcast_transaction(self, Tran)
                
        for Tran in self.Queue:
            self.Queue.remove(Tran)
            self.Tangle.append(Tran)
            if(Tran.is_tip()):
                self.TipsSet.append(Tran)
            for Parent in Tran.Parents:
                Parent.Children.append(Tran)
                if Parent in self.TipsSet:
                    self.TipsSet.remove(Child)
                else:
                    continue
            self.Network.broadcast_transaction(self, Tran)
                
    def add_to_queue(self, Tran):
        if (Tran not in Queue) and (Tran not in Tangle):
            Queue.append(Tran)
    
        
class Packet:
    
    def __init__(self, Tran, StartTime):
        self.Tran = Tran
        self.StartTime = StartTime
        
    def get_start_time(self):
        return self.StartTime
    
    
    def get_transaction(self):
        return self.Tran
    

class CommChannel:
    
    def __init__(self, TxNode, RxNode, Delay):
        self.TxNode = TxNode
        self.RxNode = RxNode
        self.Delay = Delay
        self.Packets = []
    
    def send_packet(self, Tran, Time):
        Packets.append(Packet(Tran, Time))
    
    def transmit_packets(self, Time):
        for Packet in self.Packets:
            if(Time>=Packet.get_start_time()+self.Delay):
                self.deliver_packet(Packet)
            
    def deliver_packet(self, Packet):
        RxNode.add_to_queue(Packet.get_transaction())
        self.Packets.remove(Packet)
        
class Network:
    
    def __init__(self, DelayMatrix):
        self.D = DelayMatrix
        self.Lambdas = Lambdas
        self.Nodes = []
        self.CommChannels = []
        
        for i in range(np.size(D,1)):
            Nodes.append(Node(self, Lambdas[i]))
            
        for i in range(np.size(D,1)):
            RowList = []
            for j, Delay in enumerate(np.nonzero(D[i,:])):
                Rowlist.append(CommChannel(Nodes[i],Nodes[j],Delay))
            CommChannels.append(RowList)
            
    def broadcast_transaction(self, Node, Tran, Time):
        for CommChannel in self.CommChannels[Nodes.index(Node)]:
            CommChannel.send_packet(Tran, Time)
        
    def update_comm_channels(self, Time):
        for CCs in CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time)
    
    def process_transactions(self, Time):
        for Node in self.Nodes:
            for i in range(np.random.poisson(Node.Lambda)):
                Parents = Node.select_tips()
                Node.TempTransactions.append(Transaction(Time, Parents, Node))
            Node.add_transactions(Time)
            
    