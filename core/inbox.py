from .global_params import *
from . import message as msg
import numpy as np

class Inbox:
    """
    Object for holding packets in different channels corresponding to different nodes
    """
    def __init__(self, Node):
        self.Node = Node
        self.AllPackets = [] # Inbox_m
        self.AllReadyPackets = []
        self.AllNotReadyPackets = []
        self.Packets = [[] for NodeID in range(NUM_NODES)] # Inbox_m(i)
        self.ReadyPackets = [[] for NodeID in range(NUM_NODES)]
        self.NotReadyPackets = [[] for NodeID in range(NUM_NODES)]
        self.Work = np.zeros(NUM_NODES)
        self.MsgIDs = []
        self.RRNodeID = np.random.randint(NUM_NODES) # start at a random node
        self.Deficit = np.zeros(NUM_NODES)
        self.Scheduled = []
        self.Avg = 0
        self.RequestedMsgIDs = []
        self.DroppedPackets = {}

    def update_ready(self):
        """
        Needs to be updated to only check when needed
        """
        for pkt in self.AllPackets:
            if self.is_ready(pkt):
                if pkt not in self.AllReadyPackets:
                    self.AllReadyPackets.append(pkt)
                    self.ReadyPackets[pkt.Data.NodeID].append(pkt)
            else:
                if pkt not in self.AllNotReadyPackets:
                    self.AllNotReadyPackets.append(pkt)
                    self.NotReadyPackets[pkt.Data.NodeID].append(pkt)
            
        
    def is_ready(self, Pkt):
        for pID,p in Pkt.Data.Parents.items():
            if not p.Eligible:# and not p.Confirmed:
                return False
        return True
    
    def add_packet(self, Packet):
        Msg = Packet.Data
        assert isinstance(Msg, msg.Message)
        NodeID = Msg.NodeID
        if Msg.Index in self.RequestedMsgIDs:
            if self.Packets[NodeID]:
                Packet.EndTime = self.Packets[NodeID][0].EndTime # move packet to the front of the queue
            self.RequestedMsgIDs = [msgID for msgID in self.RequestedMsgIDs if msgID != Msg.Index]
            self.Packets[NodeID].insert(0,Packet)
        else:
            self.Packets[NodeID].append(Packet)
        self.AllPackets.append(Packet)
        self.MsgIDs.append(Msg.Index)
        self.Work[NodeID] += Packet.Data.Work
        # check parents are eligible
        if self.is_ready(Packet):
            self.AllReadyPackets.append(Packet)
            self.ReadyPackets[NodeID].append(Packet)
        else:
            self.AllNotReadyPackets.append(Packet)
            self.NotReadyPackets[NodeID].append(Packet)
       
    def remove_packet(self, Packet):
        """
        Remove from Inbox and filtered inbox etc
        """
        if self.MsgIDs:
            if Packet in self.AllPackets:
                self.AllPackets.remove(Packet)
                if Packet in self.AllReadyPackets:
                    self.AllReadyPackets.remove(Packet)
                    self.ReadyPackets[Packet.Data.NodeID].remove(Packet)
                if Packet in self.AllNotReadyPackets:
                    self.AllNotReadyPackets.remove(Packet)
                    self.NotReadyPackets[Packet.Data.NodeID].remove(Packet)
                self.Packets[Packet.Data.NodeID].remove(Packet)
                self.MsgIDs.remove(Packet.Data.Index)  
                self.Work[Packet.Data.NodeID] -= Packet.Data.Work

    def drop_packet(self, Packet):
        self.remove_packet(Packet)
        msgID = Packet.Data.Index
        for _,p in Packet.Data.Parents.items():
            if msgID not in p.DependentChildren:
                p.DependentChildren.append(msgID)
        self.DroppedPackets[msgID] = Packet
    
    def drr_lds_schedule(self, Time):
        if self.Scheduled:
            return self.Scheduled.pop(0)
        
        Packets = [self.AllPackets]
        while Packets[0] and not self.Scheduled:
            if self.Deficit[self.RRNodeID]<MAX_WORK:
                self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
            i = 0
            while self.Packets[self.RRNodeID] and i<len(self.Packets[self.RRNodeID]):
                Packet = self.Packets[self.RRNodeID][i]
                if Packet not in Packets[0]:
                    i += 1
                    continue
                Work = Packet.Data.Work
                if self.Deficit[self.RRNodeID]>=Work and Packet.EndTime<=Time:
                    self.Deficit[self.RRNodeID] -= Work
                    # remove the message from all inboxes
                    self.remove_packet(Packet)
                    self.Scheduled.append(Packet)
                else:
                    i += 1
                    continue
            self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
        if self.Scheduled:
            return self.Scheduled.pop(0)

    def drr_ready_schedule(self, Time):
        if self.Scheduled:
            return self.Scheduled.pop(0)
        
        while self.AllReadyPackets and not self.Scheduled:
            if self.Deficit[self.RRNodeID]<MAX_WORK:
                self.Deficit[self.RRNodeID] += QUANTUM[self.RRNodeID]
            i = 0
            while self.ReadyPackets[self.RRNodeID] and i<len(self.ReadyPackets[self.RRNodeID]):
                Packet = self.ReadyPackets[self.RRNodeID][i]
                Work = Packet.Data.Work
                if self.Deficit[self.RRNodeID]>=Work and Packet.EndTime<=Time:
                    self.Deficit[self.RRNodeID] -= Work
                    # remove the message from all inboxes
                    self.remove_packet(Packet)
                    self.Scheduled.append(Packet)
                else:
                    i += 1
                    continue
            self.RRNodeID = (self.RRNodeID+1)%NUM_NODES
        if self.Scheduled:
            return self.Scheduled.pop(0) 
                    
    def fifo_schedule(self, Time):
        if self.AllPackets:
            Packet = self.AllPackets[0]
            # remove the message from all inboxes
            self.remove_packet(Packet)
            return Packet