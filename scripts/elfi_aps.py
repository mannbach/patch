from argparse import ArgumentParser
from typing import Any, Dict
import os

import elfi
import numpy as np
from netin.utils.constants import CLASS_ATTRIBUTE
from netin.graphs import Graph
from netin.models import PATCHModel
import matplotlib.pyplot as plt

from patch.constants import PATH_APS, PATH_POSTERIORS, PATH_PLOTS, N_SAMPLES, N_NODES_SIM, L_LFM_GLOBAL, L_LFM_LOCAL
from patch.elfi import elfi_patch, compute_m, ELFISummaryFunctions
from patch.empirical import read_graph

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("Aggregate Statistics")
    ap.add_argument("--path-aps", "-pg",
                    default=PATH_APS, type=str)
    ap.add_argument("--path-results", "-pr",
                    default=PATH_POSTERIORS, type=str)
    ap.add_argument("--n-processes", default=1, type=int)

    ap.add_argument("-lfm-global", type=str, choices=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc", type=str, choices=L_LFM_LOCAL)

    d_a = ap.parse_args()

    return d_a

def plot_results(sample):
    plt.hist2d(
        x=sample.samples['h'], y=sample.samples['tau'],
        bins=20, range=[[0, 1], [0, 1]], density=True)

    plt.axhline(np.mean(sample.samples['tau']), color="red")
    plt.axvline(np.mean(sample.samples['h']), color="red")
    # plt.legend()

    plt.xlabel('$h$')
    plt.ylabel('$\\tau$')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_PLOTS, f'joint_aps_n-samples-{N_SAMPLES}_rej.pdf'))

def main():
    print("ELFI APS\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing', num_processes=args.n_processes)

    rng = np.random.RandomState(0)
    h_prior = elfi.Prior('uniform', 0, 1, name="h")
    tau_prior = elfi.Prior('uniform', 0, 1, name="tau")

    print(f"Reading APS graph from `{args.path_aps}`...")
    graph_aps = read_graph(folder=args.path_aps)

    nodes_min = graph_aps.get_node_class(CLASS_ATTRIBUTE)
    summary_aps = {
        k: f(graph_aps)\
            for k, f in ELFISummaryFunctions()._asdict().items()
    }
    print("Summary statistics:")
    print(summary_aps)

    m = max(1, compute_m(graph_empirical=graph_aps))
    f_m = np.mean(nodes_min)

    print(f"Creating simulator (`N={N_NODES_SIM}, m={m},f_m={f_m:.2f}`)...")
    simulator = elfi.Simulator(
        elfi.tools.vectorize(elfi_patch, dtype=False),
        N_NODES_SIM, f_m, m,
        args.lfm_global, args.lfm_tc,
        h_prior, tau_prior,
        observed=graph_aps)
    summary_f = (
        elfi.Summary(elfi.tools.vectorize(f), simulator)
        for k, f in ELFISummaryFunctions()._asdict().items()
    )
    distance = elfi.Distance('cosine', *summary_f)
    rejection = elfi.Rejection(distance)

    print("Running rejection sampling (this might take a while)...")
    sample = rejection.sample(n_samples=N_SAMPLES)

    print("Summary results:")
    print(sample.summary())

    plot_results(sample)

    file_posteriors = os.path.join(
        args.path_results,
        f"lfm-g-{args.lfm_g}_lfm-t-{args.lfm_t}_posteriors.npz")
    print(f"Saved posteriors to `{file_posteriors}`...")
    np.savez(
        file=file_posteriors,
        h=sample.samples["h"],
        tau=sample.samples["tau"])

if __name__ == "__main__":
    main()
