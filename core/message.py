from .global_params import *
from copy import copy

class Message:
    """
    Object to simulate a transaction and its edges in the DAG
    """
    def __init__(self, IssueTime, Parents, Node, Network, Work=0, Index=None, VisibleTime=None, Milestone=False):
        self.IssueTime = IssueTime
        self.VisibleTime = VisibleTime
        self.Parents = Parents
        self.DependentChildren = []
        self.Network = Network
        self.Index = Network.MsgIndex
        self.Milestone = Milestone
        Network.InformedNodes[self.Index] = []
        Network.ScheduledNodes[self.Index] = []
        Network.ConfirmedNodes[self.Index] = []
        self.Work = Work
        self.CWeight = Work
        self.LastCWUpdate = self
        self.Dropped = False
        if Node:
            self.Solid = False
            self.NodeID = Node.NodeID # signature of issuing node
            self.Eligible = False
            self.Confirmed = False
            self.EligibleTime = None
            Network.MsgIssuer[Network.MsgIndex] = Node.NodeID
        else: # genesis
            self.Solid = True
            self.NodeID = 0 # Genesis is issued by Node 0
            self.Eligible = True
            self.Confirmed = True
            self.EligibleTime = 0
        Network.MsgIndex += 1

    def mark_confirmed(self, Time, Node = None):
        self.Confirmed = True
        assert Node.NodeID in self.Network.ScheduledNodes[self.Index]
        assert not Node.NodeID in self.Network.ConfirmedNodes[self.Index]
        self.Network.ConfirmedNodes[self.Index].append(Node.NodeID)
        if len(self.Network.ConfirmedNodes[self.Index])==NUM_NODES:
            self.Network.Nodes[self.NodeID].UnconfMsgs.pop(self.Index)
            self.Network.Nodes[self.NodeID].ConfMsgs[self.Index] = self
            self.Network.ConfTimes[self.Index] = Time
        for _,p in self.Parents.items():
            if not p.Confirmed:
                p.mark_confirmed(Time, Node)
    
    def updateCW(self, Time, Node, updateMsg=None, Work=None):
        if updateMsg is None:
            assert Work is None
            updateMsg = self
            Work = self.Work
        else:
            assert Work is not None
            self.CWeight += Work
            if self.CWeight >= CONF_WEIGHT:
                self.mark_confirmed(Time, Node)

        self.LastCWUpdate = updateMsg
        for _,p in self.Parents.items():
            if not p.Confirmed and p.LastCWUpdate != updateMsg:
                p.updateCW(Time, Node, updateMsg, Work)
    
    def copy(self, Node):
        Msg = copy(self)
        parentIDs = [p for p in Msg.Parents]
        parents = {}
        for pID in parentIDs:
            # if we have the parents in the ledger already, include them as parents
            if pID in Node.Ledger:
                parents[pID] = Node.Ledger[pID]
                assert parents[pID].IssueTime<self.IssueTime
            elif pID in Node.SolBuffer:
                parents[pID] = Node.SolBuffer[pID].Data
                assert parents[pID].IssueTime<self.IssueTime
            else:
                parents[pID] = None
        Msg.Parents = parents
        if self.Index == 0:
            Msg.Eligible = True
            Msg.EligibleTime = 0
            Msg.Confirmed = True
            Msg.Solid = True
        else:
            Msg.Eligible = False
            Msg.EligibleTime = None
            Msg.Confirmed = False
            Msg.Solid = False
        return Msg

    def solidify(self, Node, TxNode=None, Time=None):
        solid = True
        for pID, p in self.Parents.items():
            if p is None:
                solid = False
                if pID not in Node.MissingParentIDs:
                    Node.MissingParentIDs[pID] = [self.Index]
                    Node.Network.send_data(Node, TxNode, SolRequest(pID), Time)
                else:
                    if self.Index not in Node.MissingParentIDs[pID]:
                        Node.MissingParentIDs[pID].append(self.Index)
            elif not p.Solid:
                solid = False
                if pID not in Node.MissingParentIDs:
                    Node.MissingParentIDs[pID] = [self.Index]
                else:
                    if self.Index not in Node.MissingParentIDs[pID]:
                        Node.MissingParentIDs[pID].append(self.Index)
        self.Solid = solid
        if self.Solid:
            if self.Index in Node.MissingParentIDs:
                for cID in Node.MissingParentIDs[self.Index]:
                    child = Node.SolBuffer[cID].Data
                    assert self.Index in child.Parents
                    child.Parents[self.Index] = self
                    assert self.IssueTime < child.IssueTime
                    child.solidify(Node)

class SolRequest:
    '''
    Object to request solidification of a transaction
    '''
    def __init__(self, MsgID):
        self.MsgID = MsgID

class PruneRequest:
    """
    Request to prune issued by node "NodeID"
    """
    def __init__(self, NodeID, Forward=False):
        self.NodeID = NodeID
        self.Forward = Forward # flag to forward messages from this node or not
