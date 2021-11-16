from .global_params import *
from .inbox import Inbox
from .transaction import Transaction
from . import network as net
import numpy as np
from random import sample

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        self.TipsSet = [Genesis]
        self.Ledger = [Genesis]
        self.LedgerTranIDs = [0]
        self.Neighbours = []
        self.Network = Network
        self.Inbox = Inbox(self)
        self.NodeID = NodeID
        self.Alpha = ALPHA*REP[NodeID]/sum(REP)
        self.Lambda = NU*REP[NodeID]/sum(REP)
        self.BackOff = []
        self.LastBackOff = []
        self.LastScheduleTime = 0
        self.LastScheduleWork = 0
        self.LastIssueTime = 0
        self.LastIssueWork = 0
        self.IssuedTrans = []
        self.Undissem = 0
        self.UndissemWork = 0
        self.ServiceTimes = []
        self.ArrivalTimes = []
        self.ArrivalWorks = []
        self.InboxLatencies = []
        
    def issue_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2:
                if self.BackOff:
                    self.LastIssueTime += TAU#BETA*REP[self.NodeID]/self.Lambda
                while Time+STEP >= self.LastIssueTime + self.LastIssueWork/self.Lambda:
                    self.LastIssueTime += self.LastIssueWork/self.Lambda
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    if IOT[self.NodeID]:
                        Work = np.random.uniform(IOTLOW,IOTHIGH)
                    else:
                        Work = 1
                    self.LastIssueWork = Work
                    self.IssuedTrans.append(Transaction(self.LastIssueTime, Parents, self, self.Network, Work=Work))
            elif MODE[self.NodeID]==1:
                if IOT[self.NodeID]:
                    Work = np.random.uniform(IOTLOW,IOTHIGH)
                else:
                    Work = 1
                times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
                for t in times:
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.IssuedTrans.append(Transaction(t, Parents, self, self.Network, Work=Work))
            else:
                Work = 1
                times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
                for t in times:
                    Parents = self.select_tips()
                    #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                    self.IssuedTrans.append(Transaction(t, Parents, self, self.Network, Work=Work))
                
        # check PoW completion
        while self.IssuedTrans:
            Tran = self.IssuedTrans.pop(0)
            p = net.Packet(self, self, Tran, Tran.IssueTime)
            p.EndTime = Tran.IssueTime
            self.book(p, Tran.IssueTime)
            if MODE[self.NodeID]==3: # malicious don't consider own txs for scheduling
                self.schedule(self, Tran, Tran.IssueTime)
    
    def select_tips(self):
        """
        Implements uniform random tip selection
        """
        if len(self.TipsSet)>1:
            Selection = sample(self.TipsSet, 2)
        else:
            Selection = self.Ledger[-2:-1]
        return Selection
    
    def schedule_txs(self, Time):
        """
        schedule txs from inbox at a fixed deterministic rate NU
        """
        # sort inboxes by arrival time
        self.Inbox.AllPackets.sort(key=lambda p: p.EndTime)
        self.Inbox.ReadyPackets.sort(key=lambda p: p.EndTime)
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.EndTime)
        # process according to global rate Nu
        while self.Inbox.ReadyPackets or self.Inbox.Scheduled:
            if self.Inbox.Scheduled:
                nextSchedTime = self.LastScheduleTime+(self.LastScheduleWork/NU)
            else:
                nextSchedTime = max(self.LastScheduleTime+(self.LastScheduleWork/NU), self.Inbox.ReadyPackets[0].EndTime)
                
            if nextSchedTime<Time+STEP:             
                if SCHEDULING=='drr_lds':
                    Packet = self.Inbox.drr_lds_schedule(nextSchedTime)
                elif SCHEDULING=='fifo':
                    Packet = self.Inbox.fifo_schedule(nextSchedTime)

                if Packet is not None:
                    self.schedule(Packet.TxNode, Packet.Data, nextSchedTime)
                    # update AIMD
                    #if Packet.Data.NodeID==self.NodeID:
                    self.Network.Nodes[Packet.Data.NodeID].InboxLatencies.append(nextSchedTime-Packet.EndTime)
                    self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*sum([p.Data.Work for p in self.Inbox.Packets[self.NodeID]])
                    self.set_rate(nextSchedTime)
                    self.LastScheduleTime = nextSchedTime
                    self.LastScheduleWork = Packet.Data.Work
                    self.ServiceTimes.append(nextSchedTime)
                else:
                    break
            else:
                break
    
    def schedule(self, TxNode, Tran: Transaction, Time):
        # add to eligible set
        Tran.mark_eligible(self)

        # broadcast the packet
        self.Network.broadcast_data(self, TxNode, Tran, Time)

    def parse(self, Packet, Time):
        """
        Not fully implemented yet
        Simply makes a copy of the transaction and then calls the solidifier
        """
        self.solidify(Packet, Time)

    def solidify(self, Packet, Time):
        """
        Not implemented yet, just calls the booker
        """
        Packet.Data = Packet.Data.copy(self)
        Tran = Packet.Data
        assert isinstance(Tran, Transaction)
        Tran.solidify()
                
        self.book(Packet, Time)

    def book(self, Packet, Time):
        """
        Adds the transaction to the local copy of the ledger
        """
        # make a shallow copy of the transaction and initialise metadata
        Tran = Packet.Data
        assert isinstance(Tran, Transaction)
        if Tran.Index in self.LedgerTranIDs: # return if this tranaction is already booked
            return
        self.Ledger.append(Tran)
        self.LedgerTranIDs.append(Tran.Index)
        for p in Tran.Parents:
            p.Children.append(Tran)
        Tran.updateAW(self)
        
        if Tran.NodeID==self.NodeID:
            self.Undissem += 1
            self.UndissemWork += Tran.Work
            Tran.VisibleTime = Time
        # mark this TX as received by this node
        self.Network.InformedNodes[Tran.Index] += 1
        if self.Network.InformedNodes[Tran.Index]==NUM_NODES:
            self.Network.Throughput[Tran.NodeID] += 1
            self.Network.WorkThroughput[Tran.NodeID] += Tran.Work
            self.Network.TranDelays[Tran.Index] = Time-Tran.IssueTime
            self.Network.VisTranDelays[Tran.Index] = Time-Tran.VisibleTime
            self.Network.DissemTimes[Tran.Index] = Time
            self.Network.Nodes[Tran.NodeID].Undissem -= 1
            self.Network.Nodes[Tran.NodeID].UndissemWork -= Tran.Work

        self.enqueue(Packet, Time)
    
    def check_congestion(self, Time):
        """
        Check for rate setting
        """
        if self.Inbox.Avg>MIN_TH*REP[self.NodeID]:
            if self.Inbox.Avg>MAX_TH*REP[self.NodeID]:
                self.BackOff = True
            elif np.random.rand()<P_B*(self.Inbox.Avg-MIN_TH*REP[self.NodeID])/((MAX_TH-MIN_TH)*REP[self.NodeID]):
                self.BackOff = True
            
    def set_rate(self, Time):
        """
        Additively increase or multiplicatively decrease lambda
        """
        if MODE[self.NodeID]>0:
            if MODE[self.NodeID]==2 and Time>=START_TIMES[self.NodeID]: # AIMD starts after 1 min adjustment
                # if wait time has not passed---reset.
                if self.LastBackOff:
                    if Time < self.LastBackOff + TAU:#BETA*REP[self.NodeID]/self.Lambda:
                        self.BackOff = False
                        return
                # multiplicative decrease or else additive increase
                if self.BackOff:
                    self.Lambda = self.Lambda*BETA
                    self.BackOff = False
                    self.LastBackOff = Time
                else:
                    self.Lambda += self.Alpha
            elif MODE[self.NodeID]<3: #honest active
                self.Lambda = NU*REP[self.NodeID]/sum(REP)
            else: # malicious
                self.Lambda = 3*NU*REP[self.NodeID]/sum(REP)
        else:
            self.Lambda = 0
            
    def enqueue(self, Packet, Time):
        """
        Add to inbox if not already in inbox or already eligible
        """
        if Packet.Data not in self.Inbox.TranIDs:
            if not Packet.Data.Eligible:
                self.Inbox.add_packet(Packet)
                self.ArrivalWorks.append(Packet.Data.Work)
                self.ArrivalTimes.append(Time)
                if Packet.Data.NodeID==self.NodeID:
                    #self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*len(self.Inbox.Packets[self.NodeID])
                    self.check_congestion(Time)
                '''
                Buffer Management - Drop head queue
                '''                
                if sum(self.Inbox.Work)>W_MAX:
                    ScaledWork = np.array([self.Inbox.Work[NodeID]/REP[NodeID] for NodeID in range(NUM_NODES)])
                    MalNodeID = np.argmax(ScaledWork)
                    self.Inbox.remove_packet(self.Inbox.Packets[MalNodeID][0])
                