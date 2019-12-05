# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 22:28:39 2019

@author: Pietro Ferraro
"""
import numpy as np

class Transaction:
    
    def __init__(self, ArrivalTime, Children, No = None, Signature):
        
        self.No = No
        self.ArrivalTime = ArrivalTime
        self.Children = Children
        self.Parents = []
        self.NodeSignature = Signature


class Tangle:
    
    def __init__(self, PoW = 1):
        self.TipsSet = []
        self.TempTransactions = []
        self.LedgerTransactions = []
        self.PoW = PoW
        self.Users = []
        
        self.TipsSet.append(Transaction(0, [], 0))
        self.LedgerTransactions.append(Transaction(0, [], 0))

        
    def report(self):
        
        return None
        
        
    def assign_tips(self, NumberOfSelections):
        Selection = []
        NumberOfTips= len(self.TipsSet)
        for i in range(NumberOfSelections):
            Selection.append(self.TipsSet[np.random.randint(NumberOfTips)])
        return Selection

    def add_transactions(self, Time):
       
        
        for Tran in self.TempTransactions:
            if (Tran.ArrivalTime + self.PoW) <= Time:
                
                Tran.No = Time + np.random.normal(0,0.01)
                self.TempTransactions.remove(Tran)
                self.LedgerTransactions.append(Tran)
                self.TipsSet.append(Tran)
                for Child in Tran.Children:
                    if Child in self.TipsSet:
                        Child.Parents.append(Tran)
                        self.TipsSet.remove(Child)
                    else:
                        continue
                    
                        

class FullNode(Tangle):
    
    pass
    

class User:

    def __init__(self, Id):
        
        self.Id = Id
        self.NodesList = {}
        self.IssuedTransactions = []

    def issue_transaction(self, Node, NSelections, Time):
        
        if Node in self.NodesList:
            self.NodesList[Node] += 1 
        else:
            self.NodesList[Node] = 1

        Children = Node.assign_tips(NSelections)
        Node.TempTransactions.append(Transaction(Time, Children))
        
        
class Network:
    
    pass