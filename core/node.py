from .global_params import *
from .inbox import Inbox
from .transaction import PruneRequest, Transaction
from . import network as net
import numpy as np
from random import sample

class Node:
    """
    Object to simulate an IOTA full node
    """
    def __init__(self, Network, NodeID, Genesis, PoWDelay = 1):
        g = Genesis.copy(self)
        self.TipsSet = [g]
        self.NodeTipsSet = [[] for _ in range(NUM_NODES)]
        self.NodeTipsSet[0].append(g)
        self.Ledger = {0: g}
        self.Neighbours = []
        self.NeighbForward = [] # list for each neighbour of NodeIDs to forward to this neighbour
        self.NeighbRx = [] # list for each node of neighbours responsible for forwarding each NodeID to self
        self.Network = Network
        self.Inbox = Inbox(self)
        self.NodeID = NodeID
        self.Alpha = ALPHA*REP[NodeID]/sum(REP)
        self.Lambda = NU*REP[NodeID]/sum(REP)
        self.LambdaD = MODE[NodeID]*NU*REP[NodeID]/sum(REP)
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
        self.DroppedPackets = [[] for NodeID in range(NUM_NODES)]
        self.TranPool = []
        
    def issue_txs(self, Time):
        """
        Create new TXs at rate lambda and do PoW
        """
        if self.LambdaD:
            if IOT[self.NodeID]:
                Work = np.random.uniform(IOTLOW,IOTHIGH)
            else:
                Work = 1
            times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.LambdaD/Work)))
            for t in times:
                Parents = []
                self.TranPool.append(Transaction(t, Parents, self, self.Network, Work=Work))

        if MODE[self.NodeID]<3:
            if self.BackOff:
                self.LastIssueTime += TAU#BETA*REP[self.NodeID]/self.Lambda
            while Time+STEP >= self.LastIssueTime + self.LastIssueWork/self.Lambda and self.TranPool:
                OldestTranTime = self.TranPool[0].IssueTime
                if OldestTranTime>Time+STEP:
                    break
                self.LastIssueTime = max(OldestTranTime, self.LastIssueTime+self.LastIssueWork/self.Lambda)
                Tran = self.TranPool.pop(0)
                if SELECT_TIPS=='issue':
                    Tran.Parents = self.select_tips(Time)
                else:
                    Tran.Parents = []
                Tran.IssueTime = self.LastIssueTime
                self.LastIssueWork = Tran.Work
                self.IssuedTrans.append(Tran)

        else:
            Work = 1
            times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
            for t in times:
                Parents = self.select_tips(Time)
                #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                self.IssuedTrans.append(Transaction(t, Parents, self, self.Network, Work=Work))
                
        # check PoW completion
        while self.IssuedTrans:
            Tran = self.IssuedTrans.pop(0)
            p = net.Packet(self, self, Tran, Tran.IssueTime)
            p.EndTime = Tran.IssueTime
            self.solidify(p, Tran.IssueTime) # solidify then book this transaction
            if MODE[self.NodeID]==3: # malicious don't consider own txs for scheduling
                self.schedule(self, Tran, Tran.IssueTime)
    
    def remove_old_tips(self):
        """
        Removes old tips from the tips set
        """
        pass
    
    def select_tips(self, Time):
        """
        Implements uniform random tip selection with/without fishing depending on param setting.
        """
        done = False
        while not done:
            done = True
            if len(self.TipsSet)>1:
                ts = self.TipsSet
                Selection = sample(ts, k=2)
                for tip in Selection:
                    if FISHING and (tip.IssueTime < Time - TSC):
                        self.TipsSet.remove(tip)
                        self.NodeTipsSet[tip.NodeID].remove(tip)
                        done = False
            else:
                eligibleLedger = [tran for _,tran in self.Ledger.items() if tran.Eligible] 
                if len(eligibleLedger)>1:
                    Selection = sample(eligibleLedger, k=2)
                else:
                    Selection = eligibleLedger # genesis
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
        assert not Tran.Eligible
        Tran.mark_eligible(self)
        Tran.EligibleTime = Time
        # broadcast the packet
        self.forward(TxNode, Tran, Time)

    def forward(self, TxNode, Tran, Time):
        for i, neighb in enumerate(self.Neighbours):
            if neighb == TxNode:
                continue
            if Tran.NodeID in self.NeighbForward[i]:
                self.Network.send_data(self, neighb, Tran, Time)

    def parse(self, Packet, Time):
        """
        Not fully implemented yet
        """
        if Packet.Data.Index in self.Ledger: # return if this tranaction is already booked
            if PRUNING and Time>START_TIMES[self.NodeID]:
                if Packet.TxNode in self.NeighbRx[Packet.Data.NodeID]: # if this node is still responsible for delivering this traffic
                    if len(self.NeighbRx[Packet.Data.NodeID])>REDUNDANCY: # if more neighbours than required are responsible too, then remove this transmitting node
                        p = PruneRequest(Packet.Data.NodeID)
                        self.Network.send_data(self, Packet.TxNode, p, Time)
                        self.NeighbRx[Packet.Data.NodeID].remove(Packet.TxNode)
            return
        Packet.Data = Packet.Data.copy(self)
        Tran = Packet.Data
        assert isinstance(Tran, Transaction)
        self.solidify(Packet, Time)

    def solidify(self, Packet, Time):
        """
        Not implemented yet, just calls the booker

        Tran = Packet.Data
        Tran.solidify(self)
        """
        self.book(Packet, Time)

    def book(self, Packet, Time):
        """
        Adds the transaction to the local copy of the ledger
        """
        # make a shallow copy of the transaction and initialise metadata
        Tran = Packet.Data
        assert isinstance(Tran, Transaction)
        assert Tran.Index not in self.Ledger
        self.Ledger[Tran.Index] = Tran
        for p in Tran.Parents:
            p.Children.append(Tran)
        Tran.updateAW(self)
        
        
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
        if Tran.NodeID==self.NodeID:
            self.Undissem += 1
            self.UndissemWork += Tran.Work
            Tran.VisibleTime = Time
            if MODE[self.NodeID]==3:
                return # don't enqueue own transactions if malicious

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
        if MODE[self.NodeID]>=0:
            if Time>=START_TIMES[self.NodeID]: # AIMD starts after 1 min adjustment
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
                elif self.Lambda<2*self.LambdaD:
                    self.Lambda += self.Alpha
            elif MODE[self.NodeID]<3: #honest active
                self.Lambda = NU*REP[self.NodeID]/sum(REP)
            else: # malicious
                self.Lambda = 3*NU*REP[self.NodeID]/sum(REP)
            
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
                if sum(self.Inbox.Work)>MAX_BUFFER:
                    ScaledWork = np.array([self.Inbox.Work[NodeID]/REP[NodeID] for NodeID in range(NUM_NODES)])
                    MalNodeID = np.argmax(ScaledWork)
                    packet = self.Inbox.Packets[MalNodeID][0]
                    self.Inbox.remove_packet(packet)
                    self.DroppedPackets[MalNodeID].append(packet)

    def prune(self, TxNode, NodeID, Forward):
        neighbID = self.Neighbours.index(TxNode)
        if not Forward:
            if NodeID in self.NeighbForward[neighbID]:
                self.NeighbForward[neighbID].remove(NodeID)