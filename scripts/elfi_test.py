from typing import Any, Dict
from argparse import ArgumentParser
import os
from itertools import product

import elfi
import numpy as np
from netin.utils.constants import CLASS_ATTRIBUTE

from patch.constants import (
    PATH_POSTERIORS,
    N_SAMPLES, N_NODES_SIM,
    L_LFM_GLOBAL, L_LFM_LOCAL,
    F, M)
from patch.elfi import (
    elfi_patch,
    compute_m, create_elfi_simulator, register_summary_stats,
    compute_observed_summary_stats, register_sampler)
from patch.model_config import ModelConfig

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("Aggregate Statistics")

    ap.add_argument("--prefix", type=str, default="")

    ap.add_argument("--path-results", "-pr",
                    default=PATH_POSTERIORS, type=str)
    ap.add_argument("--n-processes", default=1, type=int)
    # Flag to store simulation data
    ap.add_argument("--store-sim-data", action="store_true")

    # Add h_true, tau_true and n_samples as arguments
    ap.add_argument("--h-true",
                    nargs="+", type=float)
    ap.add_argument("--tau-true",
                    nargs="+", type=float)

    ap.add_argument("-lfm-global-true", type=str, choices=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc-true", type=str, choices=L_LFM_LOCAL)

    ap.add_argument("-lfm-global", type=str, choices=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc", type=str, choices=L_LFM_LOCAL)

    d_a = ap.parse_args()

    return d_a

def create_folder_name(args, h: float, tau: float):
    return os.path.join(
        args.path_results,
        (f"{args.prefix}"
         f"lfm-g-true-{args.lfm_global_true}_lfm-t-true-{args.lfm_tc_true}_"
         f"h-true-{h}_tau-true-{tau}_"
         f"lfm-g-{args.lfm_global}_lfm-t-{args.lfm_tc}_"
         "/"))

def main():
    print("ELFI APS\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing',
                     num_processes=args.n_processes)

    np.random.seed(0)

    for h, tau in product(args.h_true, args.tau_true):
        print(f"Running for h={h}, tau={tau}")

        graph_obs, t_edges_obs = elfi_patch(
            N=N_NODES_SIM, f_m=F, m=M,
            lfm_global=args.lfm_global, lfm_tc=args.lfm_tc,
            h=h, tau=tau, random_state=0
        )
        nodes_min = graph_obs.get_node_class(CLASS_ATTRIBUTE)

        m = max(2, compute_m(graph_empirical=graph_obs))
        f_m = np.mean(nodes_min)
        print((
            f"\tSimulated observed graph with {len(graph_obs)} nodes, "
            f"{len(t_edges_obs) // 2} edges, and "
            f"`f_m={f_m:.2f}`, `m={m}`"))

        print(f"\tCreating simulator (`N={N_NODES_SIM}, m={m},f_m={f_m:.2f}`)...")
        simulator = create_elfi_simulator(
            model_config=ModelConfig(
                N=N_NODES_SIM, f_m=f_m, m=m,
                homophily=-1, tau=-1, # These will be ignored
                lfm_global=args.lfm_global, lfm_tc=args.lfm_tc),
            observed=(graph_obs, t_edges_obs))

                # Define summary statistics
        summary_f = register_summary_stats(simulator)

        print("\tComputing summary statistics for empirical graph:")
        s_observed = compute_observed_summary_stats(
            l_observed=[(graph_obs, t_edges_obs)],
            summary_f=summary_f)
        for k, v in s_observed.items():
            print(f"\t`{k}`: {v}")

        sampler = register_sampler(
            summary_f=summary_f)
        print("\tRunning sampling (this might take a while)...")
        # sample = sampler.sample(
            # N_SAMPLES, [0.7, 0.2, 0.05])
        sample = sampler.sample(
            N_SAMPLES, 5)

        print("\nAdaptive distance weights:")
        for i, weights in enumerate(sample.adaptive_distance_w):
            print(f"\tround {i + 1}, weights={weights}")

        file_posteriors = os.path.join(
            create_folder_name(args=args, h=h, tau=tau), "posteriors.npz")
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
