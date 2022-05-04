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
        self.DroppedPackets = [[] for NodeID in range(NUM_NODES)]
        self.MsgPool = []
        self.LastMilestoneTime = 0
        self.UnconfMsgs = {}
        self.ConfMsgs = {}
        self.SolBuffer = {}
        self.MissingParentIDs = {}
        self.TipBlacklist = []
        
    def issue_msgs(self, Time):
        """
        Create new msgs at rate lambda
        """
        if self.LambdaD:
            if IOT[self.NodeID]:
                Work = np.random.uniform(IOTLOW,IOTHIGH)
            else:
                Work = 1
            times = np.sort(np.random.uniform(Time, Time+STEP, np.random.poisson(STEP*self.LambdaD/Work)))
            for t in times:
                Parents = {}
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
                self.IssuedMsgs.append(Message(t, Parents, self, self.Network, Work=Work))
                
        # Issue the messages
        while self.IssuedMsgs:
            Msg = self.IssuedMsgs.pop(0)
            p = net.Packet(self, self, Msg, Msg.IssueTime)
            p.EndTime = Msg.IssueTime
            if self.NodeID==COO and Msg.IssueTime>=self.LastMilestoneTime+MILESTONE_PERIOD:
                Msg.Milestone = True
                self.LastMilestoneTime += MILESTONE_PERIOD
            self.UnconfMsgs[Msg.Index] = Msg
            self.solidify(p, Msg.IssueTime) # solidify then book this message
            if MODE[self.NodeID]==3: # malicious don't consider own msgs for scheduling
                self.schedule(self, Msg, Msg.IssueTime)
    
    def add_tip(self, tip):
        """
        Adds tip to the tips set
        """
        if tip.NodeID not in self.TipBlacklist:
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
        Implements uniform random tip selection
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
        assert len(Selection)==2 or Selection[0].Index==0
        return {s.Index: s for s in Selection}
    
    def schedule_msgs(self, Time):
        """
        schedule msgs from inbox at a fixed deterministic rate NU
        """
        # sort inboxes by timestamp
        self.Inbox.AllPackets.sort(key=lambda p: p.Data.IssueTime)
        self.Inbox.AllReadyPackets.sort(key=lambda p: p.Data.IssueTime)
        self.Inbox.AllNotReadyPackets.sort(key=lambda p: p.Data.IssueTime)
        for NodeID in range(NUM_NODES):
            self.Inbox.Packets[NodeID].sort(key=lambda p: p.Data.IssueTime)
        # process according to global rate Nu
        while self.Inbox.AllReadyPackets or self.Inbox.Scheduled:
            if self.Inbox.Scheduled:
                nextSchedTime = self.LastScheduleTime+(self.LastScheduleWork/NU)
            else:
                newestPacket = min(self.Inbox.AllReadyPackets, key = lambda p: p.EndTime)
                nextSchedTime = max(self.LastScheduleTime+(self.LastScheduleWork/NU), newestPacket.EndTime)
                
            if nextSchedTime<Time+STEP:             
                if SCHEDULING=='drr_lds':
                    Packet = self.Inbox.drr_lds_schedule(nextSchedTime)
                if SCHEDULING=='drr_ready':
                    Packet = self.Inbox.drr_ready_schedule(nextSchedTime)
                elif SCHEDULING=='fifo':
                    Packet = self.Inbox.fifo_schedule(nextSchedTime)

                if Packet is not None:
                    self.schedule(Packet.TxNode, Packet.Data, nextSchedTime)
                    # update AIMD
                    self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*sum([p.Data.Work for p in self.Inbox.Packets[self.NodeID]])
                    self.set_rate(nextSchedTime)
                    self.LastScheduleTime = nextSchedTime
                    self.LastScheduleWork = Packet.Data.Work
                    self.ServiceTimes.append(nextSchedTime)
                else:
                    break
            else:
                break

    def update_tipsset(self, Msg, Time):
        """
        Tip set manager
        """
        self.Inbox.update_ready()
        if OWN_TXS or Msg.NodeID!=self.NodeID or MODE[self.NodeID]>=3:
            if MAX_TIP_AGE is None:
                self.add_tip(Msg)
            elif Time-Msg.IssueTime < MAX_TIP_AGE:
                self.add_tip(Msg)

        # if this is a malicious nodes own message and ATK_TIP_RM_PARENTS is False, then don't remove the tips it selected as parents
        if MODE[self.NodeID]>=3 and Msg.NodeID==self.NodeID and not ATK_TIP_RM_PARENTS:
            pass
        else:
            # remove parents from tip set
            for _,p in Msg.Parents.items():
                if p in self.TipsSet:
                    self.remove_tip(p)
                else:
                    continue

        # check tip set size and remove oldest if too large. Malicious nodes don't prune tip set if ATK_MAX_SIZE is False.
        if L_MAX is not None:
            if len(self.TipsSet)>L_MAX and not (MODE[self.NodeID]==3 and not ATK_TIP_MAX_SIZE):
                oldestTip = min(self.TipsSet, key = lambda tip:tip.IssueTime)
                self.remove_tip(oldestTip)
    
    def schedule(self, TxNode, Msg: Message, Time):
        assert self.NodeID in self.Network.InformedNodes[Msg.Index]
        assert not self.NodeID in self.Network.ScheduledNodes[Msg.Index]
        self.Network.ScheduledNodes[Msg.Index].append(self.NodeID)
        if len(self.Network.ScheduledNodes[Msg.Index])==NUM_NODES:
            self.Network.Scheduled[Msg.NodeID] += 1
        # add to eligible set
        assert not Msg.Eligible
        assert Msg.Index in self.Ledger
        if CONF_TYPE=='CW':
            Msg.updateCW(Time, self)
        # if message is a milestone, mark its past cone as confirmed
        if CONF_TYPE=='Coo' and Msg.Milestone:
            Msg.mark_confirmed(Time, self)
        Msg.Eligible = True
        self.update_tipsset(Msg, Time)
        Msg.EligibleTime = Time
        # check if this message has dependent children that were dropped
        for childIndex in Msg.DependentChildren:
            if childIndex in self.Inbox.DroppedPackets:
                Packet = self.Inbox.DroppedPackets.pop(childIndex)
                self.enqueue(Packet)
        # broadcast the packet
        self.forward(TxNode, Msg, Time)

    def forward(self, TxNode, Msg, Time):
        """
        By default, nodes forward to all neighbours except the one they received the TX from.
        Multirate attackers select one nighbour at random to forward their own messages to.
        """
        if self.NodeID==Msg.NodeID and MODE[self.NodeID]==3 and ATK_RAND_FORWARD: # multirate attacker
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
        if Packet.Data.Index in self.Ledger or Packet.Data.Index in self.SolBuffer: # return if this tranaction is already booked or in solidification buffer
            if PRUNING and Time>START_TIMES[self.NodeID]:
                if Packet.TxNode in self.NeighbRx[Packet.Data.NodeID]: # if this node is still responsible for delivering this traffic
                    if len(self.NeighbRx[Packet.Data.NodeID])>REDUNDANCY: # if more neighbours than required are responsible too, then remove this transmitting node
                        p = PruneRequest(Packet.Data.NodeID)
                        self.Network.send_data(self, Packet.TxNode, p, Time)
                        self.NeighbRx[Packet.Data.NodeID].remove(Packet.TxNode)
            return

        # make a shallow copy of the message and initialise metadata
        Packet.Data = Packet.Data.copy(self)
        Msg = Packet.Data
        assert isinstance(Msg, Message)
        self.solidify(Packet, Time)

    def solidify(self, Packet, Time):
        """
        Not implemented yet, just calls the booker
        """
        self.SolBuffer[Packet.Data.Index] = Packet
        
        Msg = Packet.Data
        Msg.solidify(self, Packet.TxNode, Time)
        for msg in list(self.SolBuffer.keys()):
            assert msg not in self.Ledger
            if self.SolBuffer[msg].Data.Solid:
                #self.SolBuffer[msg].EndTime = Time
                pkt = self.SolBuffer.pop(msg)
                self.book(pkt, Time)


    def book(self, Packet, Time):
        """
        Adds the message to the local copy of the ledger
        """
        Msg = Packet.Data
        assert isinstance(Msg, Message) and Msg.Index not in self.Ledger and Msg.Index not in self.SolBuffer
        self.Ledger[Msg.Index] = Msg
        
        # mark this TX as received by this node
        assert not self.NodeID in self.Network.InformedNodes[Msg.Index]
        self.Network.InformedNodes[Msg.Index].append(self.NodeID)
        if len(self.Network.InformedNodes[Msg.Index])==NUM_NODES:
            self.Network.Disseminated[Msg.NodeID] += 1
            self.Network.WorkDisseminated[Msg.NodeID] += Msg.Work
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
    
    def check_congestion(self):
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
            
    def enqueue(self, Packet, Time=None):
        """
        Add to inbox if not already in inbox or already eligible
        """
        if Packet.Data not in self.Inbox.MsgIDs:
            if not Packet.Data.Eligible:
                '''
                Buffer Management - if using tail drop, don't even add the packet if it is to be dropped.
                ''' 
                '''if sum(self.Inbox.Work)+Packet.Data.Work>MAX_BUFFER and DROP_TYPE=='tail':
                    ScaledWork = np.array([self.Inbox.Work[NodeID]/REP[NodeID] for NodeID in range(NUM_NODES)])
                    MalNodeID = np.argmax(ScaledWork)
                    if MalNodeID==Packet.Data.NodeID:
                        newestPacket = self.Inbox.Packets[MalNodeID][-1]
                        if Packet.Data.IssueTime>newestPacket.Data.IssueTime:
                            self.Inbox.drop_packet(Packet)
                            self.DroppedPackets[MalNodeID].append(Packet)
                            Packet.Data.Dropped = True
                            return'''
                for pID,p in Packet.Data.Parents.items():
                    if not p.Eligible:# and not p.Confirmed:
                        if pID in self.Inbox.DroppedPackets:
                            self.Inbox.drop_packet(Packet)
                            self.DroppedPackets[Packet.Data.NodeID].append(Packet)
                            Packet.Data.Dropped = True
                            pkt = self.Inbox.DroppedPackets.pop(pID)
                            self.enqueue(pkt)
                            return
                self.Inbox.add_packet(Packet)
                self.ArrivalWorks.append(Packet.Data.Work)
                if Time is not None: # Time is none if we are re-adding a dropped packet
                    self.ArrivalTimes.append(Time)
                    if Packet.Data.NodeID==self.NodeID:
                        #self.Inbox.Avg = (1-W_Q)*self.Inbox.Avg + W_Q*len(self.Inbox.Packets[self.NodeID])
                        self.check_congestion()
                '''
                Buffer Management
                '''                
                if sum(self.Inbox.Work)>MAX_BUFFER:
                    ScaledWork = np.array([self.Inbox.Work[NodeID]/REP[NodeID] for NodeID in range(NUM_NODES)])
                    MalNodeID = np.argmax(ScaledWork)
                    if self.Inbox.NotReadyPackets[MalNodeID]:
                        self.Inbox.NotReadyPackets[MalNodeID].sort(key=lambda p: p.Data.IssueTime)
                        if DROP_TYPE=='head':
                            packet = self.Inbox.NotReadyPackets[MalNodeID][0] # Head drop
                        elif DROP_TYPE=='tail':
                            packet = self.Inbox.NotReadyPackets[MalNodeID][-1] # Tail drop
                    else:
                        self.Inbox.Packets[MalNodeID].sort(key=lambda p: p.Data.IssueTime)
                        if DROP_TYPE=='head':
                            packet = self.Inbox.Packets[MalNodeID][0] # Head drop
                        elif DROP_TYPE=='tail':
                            packet = self.Inbox.Packets[MalNodeID][-1] # Tail drop
                    self.Inbox.drop_packet(packet)
                    self.DroppedPackets[MalNodeID].append(packet)
                    packet.Data.Dropped = True

                    if TIP_BLACKLIST and (MalNodeID not in self.TipBlacklist):
                        self.TipBlacklist.append(MalNodeID)
                        for tip in self.NodeTipsSet[MalNodeID]:
                            self.remove_tip(tip)

    def prune(self, TxNode, NodeID, Forward):
        neighbID = self.Neighbours.index(TxNode)
        if not Forward:
            if NodeID in self.NeighbForward[neighbID]:
                self.NeighbForward[neighbID].remove(NodeID)