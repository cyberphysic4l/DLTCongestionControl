from .global_params import *
from copy import copy

class Transaction:
    """
    Object to simulate a transaction and its edges in the DAG
    """
    def __init__(self, IssueTime, Parents, Node, Network, Work=0, Index=None, VisibleTime=None):
        self.IssueTime = IssueTime
        self.VisibleTime = VisibleTime
        self.Children = []
        self.Parents = Parents
        self.Network = Network
        self.Index = Network.TranIndex
        Network.TranIndex += 1
        Network.InformedNodes[self.Index] = 0
        Network.ConfirmedNodes[self.Index] = 0
        self.Work = Work
        self.AWeight = Work
        self.LastAWUpdate = self
        if Node:
            self.NodeID = Node.NodeID # signature of issuing node
            self.Eligible = False
            self.Confirmed = False
        else: # genesis
            self.NodeID = []
            self.Eligible = True
            self.Confirmed = True

    def mark_confirmed(self, Node):
        self.Confirmed = True
        self.Network.ConfirmedNodes[self.Index] +=1
        self.mark_eligible(Node)

    def mark_eligible(self, Node):
        # mark this transaction as eligible and modify the tipset accordingly
        self.Eligible = True
        # add this to tipset if no eligible children
        isTip = True
        for c in self.Children:
            if c.Eligible:
                isTip = False
                break
        if isTip:
            Node.TipsSet.append(self)
        
        # remove parents from tip set
        if self.Parents:
            for p in self.Parents:
                p.Children.append(self)
                if p in Node.TipsSet:
                    Node.TipsSet.remove(p)
                else:
                    continue
    
    def updateAW(self, Node, updateTran=None, Work=None):
        if updateTran is None:
            assert Work is None
            updateTran = self
            Work = self.Work
        else:
            assert Work is not None
            self.AWeight += Work
            if self.AWeight >= CONF_WEIGHT:
                self.mark_confirmed(Node)

        self.LastAWUpdate = updateTran
        for p in self.Parents:
            if not p.Confirmed and p.LastAWUpdate != updateTran:
                p.updateAW(Node, updateTran, Work)
    
    def copy(self):
        Tran = copy(self)
        Tran.Children = []
        Tran.Eligible = False
        Tran.Confirmed = False

        return Tran

    def is_ready(self):
        for p in self.Parents:
            if not p.Eligible:
                return False
        return True

class SolRequest:
    '''
    Object to request solidification of a transaction
    '''
    def __init__(self, TranID):
        self.TranID = TranID
