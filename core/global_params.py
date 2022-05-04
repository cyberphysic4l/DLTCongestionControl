import numpy as np

# Simulation Parameters
MONTE_CARLOS = 20
SIM_TIME = 60
STEP = 0.001
# Network Parameters
NU = 250
NUM_NODES = 20
NUM_NEIGHBOURS = 4
START_TIMES = 1*np.ones(NUM_NODES)
GRAPH = 'regular'
REPDIST = 'zipf'
if REPDIST=='zipf':
    # IOTA data rep distribution - Zipf s=0.9
    REP50 = [(51)/((NodeID+1)**0.9) for NodeID in range(50)]
    REPN = [(NUM_NODES+1)/((NodeID+1)**0.9) for NodeID in range(NUM_NODES)]
    REP = [(sum(REP50)/sum(REPN))*rep for rep in REPN]
elif REPDIST=='uniform':
    # Permissioned System rep system?
    REP = np.ones(NUM_NODES, dtype=int)

# Modes: 0 = inactive, 1 = content, 2 = best-effort, 3 = malicious
MODE = [3-(NodeID+1)%4 for NodeID in range(NUM_NODES)] # multiple malicious
#MODE = [2-(NodeID)%3 for NodeID in range(NUM_NODES)] # All honest
#MODE = [1 for _ in range(NUM_NODES)] # All content (95%)
#MODE[2] = 3 # Make node 2 malicious
IOT = np.zeros(NUM_NODES)
IOTLOW = 0.5
IOTHIGH = 1
MAX_WORK = 1

# Congestion Control Parameters
# Rate Setter
ALPHA = 0.075
BETA = 0.7
TAU = 0.2
MIN_TH = 1 # W
MAX_TH = MIN_TH
P_B = 0.5 # Not used if MAX_TH==MIN_TH
W_Q = 0.1 # for exponential moving average of inbox length measurement

# Scheduler
SCHEDULING = 'drr_ready'
QUANTUM = [MAX_WORK*rep/sum(REP) for rep in REP]

# Buffer Manager
MAX_BUFFER = 500 # W_max
DROP_TYPE = 'tail'
TIP_BLACKLIST = False

# Dash visualisation
DASH = False
UPDATE_INTERVAL = 10

# Tip selection
L_MAX = None    # 'None' if no limit, otherwise max number of tips
OWN_TXS = True  # Include own txs for tip selection
MAX_TIP_AGE = 3

# Gossip optimisation
PRUNING = False
REDUNDANCY = 2

# Confirmation type
CONF_TYPE = 'CW'
## Coordinator (Coo)
MILESTONE_PERIOD = 10
COO = 0
## Cumulative Weight (CW)
CONF_WEIGHT = 200

# Attacker details
## Tip selection
ATK_TIP_MAX_SIZE = True
ATK_TIP_RM_PARENTS = True
## forwarding behaviour
ATK_RAND_FORWARD = False