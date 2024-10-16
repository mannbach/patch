from netin.models import CompoundLFM
import matplotlib.pyplot as plt

# Default values
N = 5000
M = 2
F = 0.2
N_REALIZATIONS = 100

# Parameter options
L_LFM_GLOBAL = [lfm.value for lfm in CompoundLFM]
L_LFM_LOCAL = [lfm.value for lfm in CompoundLFM]
L_HOMOPHILY = [0.01, 0.25, 0.5, 0.75, 0.99]
L_TAU = [0., 0.2, 0.5, 0.8, 1.]

# IO
PATH_GRAPHS = "./data/graphs/"
PATH_STATISTICS = "./output/aggregate_statistics.csv"
PATH_PLOTS = "./output/plots/"

# Computation
STOP_SIGNAL = -1
