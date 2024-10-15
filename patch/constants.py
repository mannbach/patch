from netin.models import CompoundLFM

# Default values
N = 5000
M = 2
F = 0.2
N_REALIZATIONS = 100

# Parameter options
L_LFM_GLOBAL = [
    CompoundLFM.UNIFORM, CompoundLFM.HOMOPHILY, CompoundLFM.PAH]
L_LFM_LOCAL = [
    CompoundLFM.UNIFORM, CompoundLFM.HOMOPHILY, CompoundLFM.PAH]
L_HOMOPHILY = [0.01, 0.25, 0.5, 0.75, 0.99]
L_TAU = [0., 0.2, 0.5, 0.8, 1.]

PATH_GRAPHS = "./data/graphs/"

STOP_SIGNAL = -1
