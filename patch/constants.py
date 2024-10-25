from netin.models import CompoundLFM
import matplotlib.pyplot as plt

# Default values
N = 5000
M = 3
F = 0.2
N_REALIZATIONS = 100

# Parameter options
L_LFM_GLOBAL = [lfm.value for lfm in CompoundLFM]
L_LFM_LOCAL = [lfm.value for lfm in CompoundLFM]
L_HOMOPHILY = [0.01, 0.25, 0.5, 0.75, 0.99]
L_TAU = [0., 0.25, 0.5, 0.75, 1.]

# IO
PATH_GRAPHS = "./data/graphs/"
PATH_STATISTICS = "./output/aggregate_statistics.csv"
PATH_PLOTS = "./output/plots/"

# Computation
STOP_SIGNAL = -1

# Plotting
COLOR_MAJ = "#2c7bb6ff"
COLOR_MIN = "#d7191cff"
PAPER_TEXT_WIDTH = 468
SIZE_FIG = (PAPER_TEXT_WIDTH / 72, PAPER_TEXT_WIDTH / 72 / 1.618)
MAP_LFM_SHORT = {
    CompoundLFM.UNIFORM.value: "U",
    CompoundLFM.HOMOPHILY.value: "H",
    CompoundLFM.PAH.value: "PAH",
}
MAP_STAT_LABEL = {
    "gini": "Gini",
    "ei": "EI-index",
    "mann_whitney": "Mann-Whitney",
}
