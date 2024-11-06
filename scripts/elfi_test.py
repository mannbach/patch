from typing import Any, Dict
from argparse import ArgumentParser
import os
import elfi
import matplotlib.pyplot as plt
import numpy as np
from netin.models import PATCHModel, CompoundLFM
from patch.statistics import compute_gini, compute_ei, compute_mann_whitney, compute_clustering_coefficient
from patch.constants import F, PATH_PLOTS

N_SAMPLES = 1000
M = 5

N_SIM=1000
N_TRUE = 5000
LFM_TC = CompoundLFM.UNIFORM
LFM_GLOBAL = CompoundLFM.PAH

H_TRUE = 0.25
TAU_TRUE = 0.75

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("Aggregate Statistics")

    # Add h_true, tau_true and n_samples as arguments
    ap.add_argument("--h-true",
                    default=H_TRUE, type=float)
    ap.add_argument("--tau-true", default=TAU_TRUE, type=float)

    d_a = ap.parse_args()

    return d_a

def simul(h: float, tau: float, N: int = None, random_state=None):
    model = PATCHModel(
        N=N_SIM if N is None else N,
        f_m=F, m=M, tau=float(tau), h_M=float(h), h_m=float(h), random_state=random_state, lfm_global=LFM_GLOBAL, lfm_tc=LFM_TC)
    g = model.simulate()
    return g

def f_gini(g):
    return compute_gini(g.degrees())
def f_ei(g):
    return (compute_ei(g) + 1) / 2
def f_mw(g):
    return compute_mann_whitney(g)
def f_ccf(g):
    return compute_clustering_coefficient(g)

def main():
    d_config = parse_args()

    elfi.set_client('multiprocessing')

    rng = np.random.RandomState(0)

    h_pr = elfi.Prior('uniform', 0, 1)
    tau_pr = elfi.Prior('uniform', 0, 1)

    g_true = simul(h=d_config.h_true, tau=d_config.tau_true, N=N_TRUE, random_state=rng)
    gini_true = f_gini(g_true)
    ei_true = f_ei(g_true)
    mw_true = f_mw(g_true)
    ccf_true = f_ccf(g_true)

    sim = elfi.Simulator(
        elfi.tools.vectorize(simul, dtype=False),
        h_pr, tau_pr,
        observed=g_true)

    s_gini = elfi.Summary(elfi.tools.vectorize(f_gini), sim)
    s_ei = elfi.Summary(elfi.tools.vectorize(f_ei), sim)
    s_mw = elfi.Summary(elfi.tools.vectorize(f_mw), sim)
    s_ccf = elfi.Summary(elfi.tools.vectorize(f_ccf), sim)

    d = elfi.Distance('cosine', s_gini, s_ei, s_mw, s_ccf)

    rej = elfi.Rejection(d)

    sample = rej.sample(n_samples=N_SAMPLES)

    print(sample.summary())

    plt.hist2d(
        x=sample.samples['h_pr'], y=sample.samples['tau_pr'],
        bins=20, range=[[0, 1], [0, 1]], density=True)

    plt.axhline(d_config.tau_true, color="red", linestyle="--", label="True")
    plt.axvline(d_config.h_true, color="red", linestyle="--")

    plt.axhline(np.mean(sample.samples['tau_pr']), color="red", label="Estimated")
    plt.axvline(np.mean(sample.samples['h_pr']), color="red")
    plt.legend()

    plt.title(
        f"$ei_{{sc}}={ei_true:.2f},gini={gini_true:.2f},mw={mw_true:.2f},ccf={ccf_true:.2f}$")
    plt.xlabel('$h$')
    plt.ylabel('$\\tau$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_PLOTS, f'joint_f-{F}_m-{M}_h-true-{d_config.h_true}_tau-true-{d_config.tau_true}_n-sampl-{N_SAMPLES}_rej.pdf'))

if __name__ == "__main__":
    main()
