from netin.models import CompoundLFM

# Default values
N = 5000
M = 3
F = 0.2
N_REALIZATIONS = 100

# Parameter options
L_LFM_GLOBAL = [CompoundLFM.HOMOPHILY.value, CompoundLFM.PAH.value]
L_LFM_LOCAL = [lfm.value for lfm in CompoundLFM]
L_HOMOPHILY = [0.01, 0.25, 0.5, 0.75, 0.99]
L_TAU = [0., 0.25, 0.5, 0.75, 1.]

# IO
PATH_GRAPHS = "./data/graphs/"
PATH_STATISTICS = "./output/stats/"
PATH_PLOTS = "./output/plots/"
PATH_APS = "./data/empirical/"
PATH_INFERENCE = "./output/inference/"
PATH_INFERENCE_VALIDATION = "./output/inference/validation/"

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
MAP_CM_H = {
    L_HOMOPHILY[0]: "#5e3c99",
    L_HOMOPHILY[1]: "#9b6dbf",
    L_HOMOPHILY[2]: "#b0b0b0",
    L_HOMOPHILY[3]: "#fdb863",
    L_HOMOPHILY[4]: "#e66101"
}

# Inference
N_SAMPLES = 1000
N_NODES_SIM = 1000
