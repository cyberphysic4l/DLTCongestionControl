from .global_params import *
from . import message as msg
from . import node
import numpy as np


class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix):
        self.A = AdjMatrix
        self.MsgIndex = 0
        self.InformedNodes = {0: [NodeID for NodeID in range(NUM_NODES)]}
        self.ScheduledNodes = {0: [NodeID for NodeID in range(NUM_NODES)]}
        self.ConfirmedNodes = {0: [NodeID for NodeID in range(NUM_NODES)]}
        self.Nodes = []
        self.CommChannels = []
        self.Disseminated = [0 for NodeID in range(NUM_NODES)]
        self.Scheduled = [0 for NodeID in range(NUM_NODES)]
        self.WorkDisseminated = [0 for NodeID in range(NUM_NODES)]
        self.MsgDelays = {}
        self.VisMsgDelays = {}
        self.DissemTimes = {}
        self.MsgIssuer = {}
        Genesis = msg.Message(0, [], [], self)
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(node.Node(self, i, Genesis))
        # Add neighbours and create list of comm channels corresponding to each node
        for i in range(np.size(self.A,1)):
            RowList = []
            for j in np.nditer(np.nonzero(self.A[i,:])):
                self.Nodes[i].Neighbours.append(self.Nodes[j])
                self.Nodes[i].NeighbForward.append([n for n in range(NUM_NODES)])
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.A[i][j]))
            self.Nodes[i].NeighbRx = [[n for n in self.Nodes[i].Neighbours] for _ in range(NUM_NODES)]
            self.CommChannels.append(RowList)
    
    def send_data(self, TxNode, RxNode, Data, Time):
        """
        Send this data (TX or back off) to a specified neighbour
        """
        cc = self.CommChannels[TxNode.NodeID][TxNode.Neighbours.index(RxNode)]
        cc.send_packet(TxNode, RxNode, Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate new messages
        """
        for Node in self.Nodes:
            Node.issue_msgs(Time)
        """
        Move packets through all comm channels
        """
        for ccs in self.CommChannels:
            for cc in ccs:
                cc.transmit_packets(Time+STEP)
        """
        Each node schedules messages in inbox
        """
        for Node in self.Nodes:
            Node.schedule_msgs(Time)
    
    def msg_latency(self, latencies, latTimes):
        for i,Msg in self.Nodes[0].Ledger.items():
            if i in self.DissemTimes and Msg.IssueTime>20:
                latencies[Msg.NodeID].append(self.DissemTimes[i]-Msg.IssueTime)
                latTimes[Msg.NodeID].append(self.DissemTimes[i])
        return latencies, latTimes

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
        self.PacketDelays = []
    
    def send_packet(self, TxNode, RxNode, Data, Time):
        """
        Add new packet to the comm channel with time of arrival
        """
        self.Packets.append(Packet(TxNode, RxNode, Data, Time))
        self.PacketDelays.append(np.random.normal(loc=self.Delay, scale=1/NU))
    
    def transmit_packets(self, Time):
        """
        Move packets through the comm channel, simulating delay
        """
        if self.Packets:
            for Packet in self.Packets:
                i = self.Packets.index(Packet)
                if(self.Packets[i].StartTime+self.PacketDelays[i]<=Time):
                    self.deliver_packet(self.Packets[i], self.Packets[i].StartTime+self.PacketDelays[i])
        else:
            pass
            
    def deliver_packet(self, Packet, Time):
        """
        When packet has arrived at receiving node, process it
        """
        Packet.EndTime = Time
        if isinstance(Packet.Data, msg.Message):
            # if this is a message, add the Packet to Inbox
            self.RxNode.parse(Packet, Time)
        elif isinstance(Packet.Data, msg.SolRequest):
            # else if this is a solidification request, retrieve the message and send it back
            MsgID = Packet.Data.MsgID
            Msg = self.RxNode.Ledger[MsgID]
            self.RxNode.Network.send_data(self.RxNode, self.TxNode, Msg, Time)
        elif isinstance(Packet.Data, msg.PruneRequest):
            Packet.RxNode.prune(Packet.TxNode, Packet.Data.NodeID, Packet.Data.Forward)
        PacketIndex = self.Packets.index(Packet)
        self.Packets.remove(Packet)
        del self.PacketDelays[PacketIndex]

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