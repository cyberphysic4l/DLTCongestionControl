from .global_params import *
from . import transaction as tran
from . import node
import numpy as np


class Network:
    """
    Object containing all nodes and their interconnections
    """
    def __init__(self, AdjMatrix):
        self.A = AdjMatrix
        self.TranIndex = 0
        self.InformedNodes = {}
        self.ConfirmedNodes = {}
        self.Nodes = []
        self.CommChannels = []
        self.Throughput = [0 for NodeID in range(NUM_NODES)]
        self.WorkThroughput = [0 for NodeID in range(NUM_NODES)]
        self.TranDelays = {}
        self.VisTranDelays = {}
        self.DissemTimes = {}
        Genesis = tran.Transaction(0, [], [], self)
        # Create nodes
        for i in range(np.size(self.A,1)):
            self.Nodes.append(node.Node(self, i, Genesis))
        # Add neighbours and create list of comm channels corresponding to each node
        for i in range(np.size(self.A,1)):
            RowList = []
            for j in np.nditer(np.nonzero(self.A[i,:])):
                self.Nodes[i].Neighbours.append(self.Nodes[j])
                RowList.append(CommChannel(self.Nodes[i],self.Nodes[j],self.A[i][j]))
            self.CommChannels.append(RowList)
    
    def send_data(self, TxNode, RxNode, Data, Time):
        """
        Send this data (TX or back off) to a specified neighbour
        """
        CC = self.CommChannels[TxNode.NodeID][TxNode.Neighbours.index(RxNode)]
        CC.send_packet(TxNode, RxNode, Data, Time)
        
    def broadcast_data(self, TxNode, LastTxNode, Data, Time):
        """
        Send this data (TX or back off) to all neighbours
        """
        for i, CC in enumerate(self.CommChannels[self.Nodes.index(TxNode)]):
            # do not send to this node if it was received from this node
            if isinstance(Data, tran.Transaction):
                if LastTxNode==TxNode.Neighbours[i]:
                    continue
            CC.send_packet(TxNode, TxNode.Neighbours[i], Data, Time)
        
    def simulate(self, Time):
        """
        Each node generate new transactions
        """
        for Node in self.Nodes:
            Node.issue_txs(Time)
        """
        Move packets through all comm channels
        """
        for CCs in self.CommChannels:
            for CC in CCs:
                CC.transmit_packets(Time+STEP)
        """
        Each node schedule transactions in inbox
        """
        for Node in self.Nodes:
            Node.schedule_txs(Time)
    
    def tran_latency(self, latencies, latTimes):
        for i,Tran in self.Nodes[0].Ledger.items():
            if i in self.DissemTimes and Tran.IssueTime>20:
                latencies[Tran.NodeID].append(self.DissemTimes[i]-Tran.IssueTime)
                latTimes[Tran.NodeID].append(self.DissemTimes[i])
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
        if isinstance(Packet.Data, tran.Transaction):
            # if this is a transaction, add the Packet to Inbox
            self.RxNode.parse(Packet, Time)
        elif isinstance(Packet.Data, tran.SolRequest):
            # else if this is a solidification request, retrieve the transaction and send it back
            TranID = Packet.Data.TranID
            Tran = self.RxNode.Ledger[TranID]
            self.RxNode.Network.send_data(self.RxNode, self.TxNode, Tran, Time)
        else:
            # else this is a back off notification
            self.RxNode.process_cong_notif(Packet, Time)
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