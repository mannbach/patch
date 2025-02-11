from argparse import ArgumentParser
from typing import Any, Dict
import os
from itertools import product

import elfi
import numpy as np
from netin.utils.constants import CLASS_ATTRIBUTE

from patch.constants import (
    PATH_APS, PATH_INFERENCE,
    N_SAMPLES, N_NODES_SIM,
    L_LFM_GLOBAL, L_LFM_LOCAL,
    N_ROUNDS)
from patch.elfi import (
    compute_m, create_elfi_simulator,
    register_summary_stats_functions,
    register_sampler)
from patch.empirical import read_graph
from patch.model_config import ModelConfig

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
                    default=PATH_INFERENCE, type=str)
    ap.add_argument("--decades", type=int, nargs="+")
    ap.add_argument("--n-processes", default=1, type=int)

    ap.add_argument("-lfm-global", type=str, choices=L_LFM_GLOBAL, nargs="+", default=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc", type=str, choices=L_LFM_LOCAL, nargs="+", default=L_LFM_LOCAL)

    # Flag to store simulation data
    ap.add_argument("--store-sim-data", action="store_true")
    ap.add_argument("--n-rounds-smc", default=N_ROUNDS, type=int)

    ap.add_argument("--prefix", type=str, default="")

    d_a = ap.parse_args()

    return d_a

def create_folder_name(
        args, lfm_global: str, lfm_tc: str, decade: int):
    return os.path.join(
        args.path_results,
        f"{args.prefix}lfm-g-{lfm_global}_lfm-t-{lfm_tc}_d-{decade}/")

def main():
    print("ELFI APS\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing',
                     num_processes=args.n_processes)

    np.random.seed(0)

    for decade in args.decades:
        print(f"\nRunning for decade `{decade}`...")
        print(f"Reading APS graph from `{args.path_aps}`...")
        graph_aps, t_edges_aps = read_graph(
            folder=args.path_aps, decade=decade)
        nodes_min = graph_aps.get_node_class(CLASS_ATTRIBUTE)

        m = max(2, compute_m(graph_empirical=graph_aps))
        f_m = np.mean(nodes_min)
        print((
            f"Read graph with {len(graph_aps)} nodes, "
            f"{len(t_edges_aps) // 2} edges, and "
            f"`f_m={f_m:.2f}`, `m={m}`"))
        for lfm_global_inf, lfm_tc_inf in product(args.lfm_global, args.lfm_tc):
            print((
                f"\t\tCreating simulator (`N={N_NODES_SIM}, m={m}, f_m={f_m:.2f}, "
                f"lfm_global={lfm_global_inf}, lfm_tc={lfm_tc_inf}`)..."))
            try:
                model_config = ModelConfig(
                    N=N_NODES_SIM, f_m=f_m, m=m,
                    realization=-1, # Will be ignored
                    homophily=-1, tau=-1, # Will be ignored
                    lfm_global=lfm_global_inf, lfm_tc=lfm_tc_inf)
            except ValueError as e:
                print(f"\tSkipping combination: {e}")
                continue
            simulator = create_elfi_simulator(
                model_config=model_config)

            folder_name = create_folder_name(
                args,
                lfm_global=lfm_global_inf,
                lfm_tc=lfm_tc_inf,
                decade=decade)
            if not os.path.exists(folder_name):
                print(f"Creating folder `{folder_name}`...")
                os.makedirs(folder_name)

            # Define summary statistics
            summary_f = register_summary_stats_functions(
                simulator, l_observations=[(graph_aps, t_edges_aps)])

            arraypool_summaries = elfi.ArrayPool(
                [summary.name for summary in summary_f],
                prefix=folder_name,
                name="summary_stats_sim")\
                    if args.store_sim_data else None
            if arraypool_summaries:
                print(f"Storing simulation data to `{arraypool_summaries.prefix}`...")

            sampler = register_sampler(
                summary_sim=summary_f,
                pool=arraypool_summaries)
            print("Running sampling (this might take a while)...")
            # sample = sampler.sample(
                # N_SAMPLES, [0.7, 0.2, 0.05])
            sample = sampler.sample(
                N_SAMPLES, args.n_rounds_smc)

            print("Summary results:")
            print(sample.summary())

            print("\nAdaptive distance weights:")
            for i, weights in enumerate(sample.adaptive_distance_w):
                print(f"\tround {i + 1}, weights={weights}")

            file_posteriors = os.path.join(
                folder_name, "posteriors.npz")
            print(f"Saved posteriors to `{file_posteriors}`...")
            np.savez(
                file=file_posteriors,
                h=sample.samples["h"],
                tau=sample.samples["tau"],
                discrepancies=sample.discrepancies,
                distance_weights=sample.adaptive_distance_w,
                **{summary.name: summary.observed for summary in summary_f})

if __name__ == "__main__":
    main()
