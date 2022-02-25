from .global_params import *
from copy import copy

class Message:
    """
    Object to simulate a transaction and its edges in the DAG
    """
    def __init__(self, IssueTime, Parents, Node, Network, Work=0, Index=None, VisibleTime=None):
        self.IssueTime = IssueTime
        self.VisibleTime = VisibleTime
        self.Children = []
        self.Parents = Parents
        self.Network = Network
        self.Index = Network.MsgIndex
        Network.InformedNodes[self.Index] = 0
        Network.ConfirmedNodes[self.Index] = 0
        self.Work = Work
        self.AWeight = Work
        self.LastAWUpdate = self
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

    def mark_confirmed(self, Node):
        self.Confirmed = True
        self.Network.ConfirmedNodes[self.Index] +=1
    
    def updateAW(self, Node, updateMsg=None, Work=None):
        if updateMsg is None:
            assert Work is None
            updateMsg = self
            Work = self.Work
        else:
            assert Work is not None
            self.AWeight += Work
            if self.AWeight >= CONF_WEIGHT:
                self.mark_confirmed(Node)

        self.LastAWUpdate = updateMsg
        for p in self.Parents:
            if not p.Confirmed and p.LastAWUpdate != updateMsg:
                p.updateAW(Node, updateMsg, Work)
    
    def copy(self, Node):
        Msg = copy(self)
        parentIDs = [p.Index for p in Msg.Parents]
        parents = []
        for pID in parentIDs:
            # if we have the parents in the ledger already, include them as parents
            if pID in Node.Ledger:
                parents.append(Node.Ledger[pID])
        Msg.Parents = parents
        childrenIDs = [c.Index for c in Msg.Children]
        children = []
        for cID in childrenIDs:
            # if children are in our ledger already, then include them (needed for solidification)
            if cID in Node.Ledger:
                children.append(Node.Ledger[cID])
        Msg.Children = children
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

    def solidify(self, Node = None):
        if len(self.Parents)>2:
            print("more than 2 parents...")
        solidParents = [p for p in self.Parents if p.Solid]
        if len(solidParents)==1:
            if self.Parents[0].Index==0: # if parent is genesis
                self.Solid = True
        if len(solidParents)==2: # if two solid parents
            self.Solid = True
        if self.Solid:
            # if we already have some children of this solid transaction, they will possibly need to be solidified too.
            for c in self.Children:
                assert isinstance(c, Message)
                if self not in c.Parents:
                    if len(c.Parents)==2:
                        print("3rd parent being added...")
                    c.Parents.append(self)
                c.solidify()


    def is_ready(self):
        eligConfParents = [p for p in self.Parents if p.Eligible or p.Confirmed]
        if len(eligConfParents)==1:
            if self.Parents[0].Index==0: # if one parent eligible/confirmed
                return True
        if len(eligConfParents)==2: # if two parents eligible/confirmed
            return True
        return False

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
