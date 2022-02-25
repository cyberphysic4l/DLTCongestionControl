from .global_params import *
from .inbox import Inbox
from .message import PruneRequest, Message
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
        if MODE[NodeID]==0:
            self.LambdaD = 0
        elif MODE[NodeID]==1:
            self.LambdaD = 0.95*self.Lambda
        else:
            self.LambdaD = 5*self.Lambda # higher than it will be allowed
        self.BackOff = []
        self.LastBackOff = []
        self.LastScheduleTime = 0
        self.LastScheduleWork = 0
        self.LastIssueTime = 0
        self.LastIssueWork = 0
        self.IssuedMsgs = []
        self.Undissem = 0
        self.UndissemWork = 0
        self.ServiceTimes = []
        self.ArrivalTimes = []
        self.ArrivalWorks = []
        self.InboxLatencies = []
        self.DroppedPackets = [[] for NodeID in range(NUM_NODES)]
        self.MsgPool = []
        
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
                self.MsgPool.append(Message(t, Parents, self, self.Network, Work=Work))

        if MODE[self.NodeID]<3:
            if self.BackOff:
                self.LastIssueTime += TAU#BETA*REP[self.NodeID]/self.Lambda
            while Time+STEP >= self.LastIssueTime + self.LastIssueWork/self.Lambda and self.MsgPool:
                OldestMsgTime = self.MsgPool[0].IssueTime
                if OldestMsgTime>Time+STEP:
                    break
                self.LastIssueTime = max(OldestMsgTime, self.LastIssueTime+self.LastIssueWork/self.Lambda)
                Msg = self.MsgPool.pop(0)
                Msg.Parents = self.select_tips(Time)
                Msg.IssueTime = self.LastIssueTime
                self.LastIssueWork = Msg.Work
                self.IssuedMsgs.append(Msg)

        else:
            Work = 1
            times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.Lambda/Work)))
            for t in times:
                Parents = self.select_tips(Time)
                #Work = np.random.uniform(AVG_WORK[self.NodeID]-0.5, AVG_WORK[self.NodeID]+0.5)
                self.IssuedMsgs.append(Message(t, Parents, self, self.Network, Work=Work))
                
        # check PoW completion
        while self.IssuedMsgs:
            Msg = self.IssuedMsgs.pop(0)
            p = net.Packet(self, self, Msg, Msg.IssueTime)
            p.EndTime = Msg.IssueTime
            self.solidify(p, Msg.IssueTime) # solidify then book this message
            if MODE[self.NodeID]>=3: # malicious don't consider own txs for scheduling
                self.schedule(self, Msg, Msg.IssueTime)
    
    def add_tip(self, tip):
        """
        Adds tip to the tips set
        """
        self.TipsSet.append(tip)
        self.NodeTipsSet[tip.NodeID].append(tip)
    
    def remove_tip(self, tip):
        """
        Removes tip from the tips set
        """
        self.TipsSet.remove(tip)
        self.NodeTipsSet[tip.NodeID].remove(tip)
    
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
            else:
                eligibleLedger = [msg for _,msg in self.Ledger.items() if msg.Eligible] 
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

    def update_tipsset(self, Msg):
        """
        Tip set manager
        """
        is_malicious = (MODE[self.NodeID]>=3 and Msg.NodeID==self.NodeID)
        
        # add to tip set if no eligible children
        isTip = True
        for c in Msg.Children:
            self.Inbox.update_ready(c)
            if c.Eligible:
                isTip = False
                break
        if isTip:
            self.add_tip(Msg)

        # if this is a malicious nodes own message, then don't remove the tips it selected as parents
        if MODE[self.NodeID]>=3 and Msg.NodeID==self.NodeID:
            pass
        else:
            # remove parents from tip set
            for p in Msg.Parents:
                if p in self.TipsSet:
                    self.remove_tip(p)
                else:
                    continue

        # check tip set size and remove oldest if too large. Malicious nodes don't prune tip set
        if len(self.TipsSet)>L_MAX:# and MODE[self.NodeID]<3:
            oldestTip = min(self.TipsSet, key = lambda tip:tip.IssueTime)
            self.remove_tip(oldestTip)
    
    def schedule(self, TxNode, Msg: Message, Time):
        # add to eligible set
        assert not Msg.Eligible
        Msg.Eligible = True
        self.update_tipsset(Msg)
        Msg.EligibleTime = Time
        # broadcast the packet
        self.forward(TxNode, Msg, Time)

    def forward(self, TxNode, Msg, Time):
        """
        By default, nodes forward to all neighbours except the one they received the TX from.
        Multirate attackers select one nighbour at random to forward their own messages to.
        """
        if self.NodeID==Msg.NodeID and MODE[self.NodeID]==4: # multirate attacker
            i = np.random.randint(NUM_NEIGHBOURS)
            self.Network.send_data(self, self.Neighbours[i], Msg, Time)
        else: # normal nodes
            for i, neighb in enumerate(self.Neighbours):
                if neighb == TxNode:
                    continue
                if Msg.NodeID in self.NeighbForward[i]:
                    self.Network.send_data(self, neighb, Msg, Time)

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
        Msg = Packet.Data
        assert isinstance(Msg, Message)
        self.solidify(Packet, Time)

    def solidify(self, Packet, Time):
        """
        Not implemented yet, just calls the booker

        Msg = Packet.Data
        Msg.solidify(self)
        """
        self.book(Packet, Time)

    def book(self, Packet, Time):
        """
        Adds the message to the local copy of the ledger
        """
        # make a shallow copy of the message and initialise metadata
        Msg = Packet.Data
        assert isinstance(Msg, Message)
        assert Msg.Index not in self.Ledger
        self.Ledger[Msg.Index] = Msg
        for p in Msg.Parents:
            p.Children.append(Msg)
        Msg.updateAW(self)
        
        
        # mark this TX as received by this node
        self.Network.InformedNodes[Msg.Index] += 1
        if self.Network.InformedNodes[Msg.Index]==NUM_NODES:
            self.Network.Throughput[Msg.NodeID] += 1
            self.Network.WorkThroughput[Msg.NodeID] += Msg.Work
            self.Network.MsgDelays[Msg.Index] = Time-Msg.IssueTime
            self.Network.VisMsgDelays[Msg.Index] = Time-Msg.VisibleTime
            self.Network.DissemTimes[Msg.Index] = Time
            self.Network.Nodes[Msg.NodeID].Undissem -= 1
            self.Network.Nodes[Msg.NodeID].UndissemWork -= Msg.Work
        if Msg.NodeID==self.NodeID:
            self.Undissem += 1
            self.UndissemWork += Msg.Work
            Msg.VisibleTime = Time
            if MODE[self.NodeID]>=3:
                return # don't enqueue own messages if malicious

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
                else:
                    self.Lambda += self.Alpha
            elif MODE[self.NodeID]<3: #honest active
                self.Lambda = NU*REP[self.NodeID]/sum(REP)
            else: # malicious
                self.Lambda = 5*NU*REP[self.NodeID]/sum(REP)
            
    def enqueue(self, Packet, Time):
        """
        Add to inbox if not already in inbox or already eligible
        """
        if Packet.Data not in self.Inbox.MsgIDs:
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