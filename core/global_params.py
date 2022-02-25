import numpy as np

# Simulation Parameters
MONTE_CARLOS = 1
SIM_TIME = 300
STEP = 0.01
# Network Parameters
NU = 20
NUM_NODES = 20
NUM_NEIGHBOURS = 4
START_TIMES = 10*np.ones(NUM_NODES)
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
#MODE = [3-(NodeID+1)%4 for NodeID in range(NUM_NODES)] # multiple malicious
MODE = [2-(NodeID)%3 for NodeID in range(NUM_NODES)] # All honest
#MODE = [1 for _ in range(NUM_NODES)] # All content (95%)
MODE[2] = 3 # Make node 2 malicious
IOT = np.zeros(NUM_NODES)
IOTLOW = 0.5
IOTHIGH = 1
MAX_WORK = 1

# Congestion Control Parameters
ALPHA = 0.075
BETA = 0.7
TAU = 2
MIN_TH = 1
MAX_TH = MIN_TH
QUANTUM = [MAX_WORK*rep/sum(REP) for rep in REP]
W_Q = 0.1
P_B = 0.5
MAX_BUFFER = 300
GRAPH = 'regular'
SCHEDULING = 'drr_lds'
CONF_WEIGHT = 100

# Dash visualisation
DASH = False
UPDATE_INTERVAL = 10

# Tip selection
L_MAX = 300

# Gossip optimisation
PRUNING = False
REDUNDANCY = 2