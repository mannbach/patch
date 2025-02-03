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
from patch.statistics import compute_group_ccf

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
    ap.add_argument("--decades", type=int, nargs="+")
    ap.add_argument("--n-processes", default=1, type=int)

    ap.add_argument("-lfm-global", type=str, choices=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc", type=str, choices=L_LFM_LOCAL)

    # Flag to store simulation data
    ap.add_argument("--store-sim-data", action="store_true")

    ap.add_argument("--prefix", type=str, default="")

    d_a = ap.parse_args()

    return d_a

def plot_results(sample, args, decade):
    plt.figure(figsize=(8, 6))
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
    file = os.path.join(create_folder_name(args, decade), "posteriors.pdf")
    print(f"Saving plot to `{file}`...")
    plt.savefig(file)

def _choose_i(data: np.ndarray, i:int):
    return data[i]

def create_folder_name(args, decade: int):
    return os.path.join(
        args.path_results,
        f"{args.prefix}lfm-g-{args.lfm_global}_lfm-t-{args.lfm_tc}_d-{decade}/")

def main():
    print("ELFI APS\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing', num_processes=args.n_processes)

    np.random.seed(0)

    for decade in args.decades:
        print(f"\nRunning for decade `{decade}`...")
        model = elfi.ElfiModel()

        if not os.path.exists(create_folder_name(args, decade)):
            print(f"Creating folder `{create_folder_name(args, decade)}`...")
            os.makedirs(create_folder_name(args, decade))

        h_prior = elfi.Prior('uniform', 0, 1, model=model, name="h")
        tau_prior = elfi.Prior('uniform', 0, 1, model=model, name="tau")

        print(f"Reading APS graph from `{args.path_aps}`...")
        graph_aps, t_edges_aps = read_graph(
            folder=args.path_aps, decade=decade)

        nodes_min = graph_aps.get_node_class(CLASS_ATTRIBUTE)

        m = max(2, compute_m(graph_empirical=graph_aps))
        f_m = np.mean(nodes_min)
        print(f"Read graph with {len(graph_aps)} nodes, {len(t_edges_aps) // 2} edges, and `f_m={f_m:.2f}`, `m={m}`")

        print(f"Creating simulator (`N={N_NODES_SIM}, m={m},f_m={f_m:.2f}`)...")
        simulator = elfi.Simulator(
            elfi.tools.vectorize(
                elfi_patch, # Simulator function
                constants=(0, 1, 2, 3, 4), # Constant arguments
                dtype=False), # Non-array dtype of simulation
            N_NODES_SIM, f_m, m,
            args.lfm_global, args.lfm_tc,
            h_prior, tau_prior,
            name="simulator", # For later reference
            observed=((graph_aps, t_edges_aps)))

        # Define summary statistics
        summary_f = [
            elfi.Summary(elfi.tools.vectorize(f), simulator, name=k)
            for k, f in ELFISummaryFunctions()._asdict().items()
        ]
        s_ccf = elfi.Summary(elfi.tools.vectorize(compute_group_ccf), simulator)
        # CCF summaries need to be flattened
        for i in range(8):
            summary_f.append(
                elfi.Summary(
                    elfi.tools.vectorize(_choose_i, constants=(1,)), # Choose the i-th element
                    s_ccf, elfi.Constant(i, model=model), # Of the CCF summary
                    name=f"ccf_{i}")
            )
        print("Computing summary statistics for empirical graph:")
        s_observed = {
            s_f.name: s_f.generate(with_values={'simulator': (graph_aps, t_edges_aps)})
            for s_f in summary_f
        }
        for k, v in s_observed.items():
            print(f"\t`{k}`: {v}")

        arraypool_summaries = elfi.ArrayPool(
            [summary.name for summary in summary_f],
            prefix=create_folder_name(args, decade),
            name="summary_stats_sim")\
                if args.store_sim_data else None
        if arraypool_summaries:
            print(f"Storing simulation data to `{arraypool_summaries.prefix}`...")

        distance = elfi.AdaptiveDistance(*summary_f)
        sampler = elfi.AdaptiveDistanceSMC(
            distance, pool=arraypool_summaries)

        print("Running rejection sampling (this might take a while)...")

        # sample = sampler.sample(
            # N_SAMPLES, [0.7, 0.2, 0.05])
        sample = sampler.sample(
            N_SAMPLES, 5)

        print("Summary results:")
        print(sample.summary())

        print("\nAdaptive distance weights:")
        for i, weights in enumerate(sample.adaptive_distance_w):
            print(f"\tround {i + 1}, weights={weights}")

        plot_results(sample, args=args, decade=decade)

        file_posteriors = os.path.join(
            create_folder_name(args, decade), "posteriors.npz")
        print(f"Saved posteriors to `{file_posteriors}`...")
        np.savez(
            file=file_posteriors,
            h=sample.samples["h"],
            tau=sample.samples["tau"],
            discrepancies=sample.discrepancies,
            distance_weights=sample.adaptive_distance_w,
            **s_observed)

if __name__ == "__main__":
    main()
