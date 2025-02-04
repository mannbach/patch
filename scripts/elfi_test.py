from typing import Any, Dict
from argparse import ArgumentParser
import os
from itertools import product

import elfi
import numpy as np
from netin.utils.constants import CLASS_ATTRIBUTE

from patch.constants import (
    PATH_INFERENCE_VALIDATION,
    N_SAMPLES, N_NODES_SIM,
    L_HOMOPHILY, L_TAU,
    L_LFM_GLOBAL, L_LFM_LOCAL,
    N_REALIZATIONS,
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
                    default=PATH_INFERENCE_VALIDATION, type=str)
    ap.add_argument("--n-processes", default=1, type=int)
    # Flag to store simulation data
    ap.add_argument("--store-sim-data", action="store_true")

    # Add h_true, tau_true and n_samples as arguments
    ap.add_argument("--h-true",
                    nargs="+", type=float, default=L_HOMOPHILY)
    ap.add_argument("--tau-true",
                    nargs="+", type=float, default=L_TAU)

    ap.add_argument("-lfm-global-true",
                    type=str, choices=L_LFM_GLOBAL,
                    nargs="+", default=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc-true",
                    type=str, choices=L_LFM_LOCAL,
                    nargs="+", default=L_LFM_LOCAL)

    ap.add_argument("-lfm-global-inf", type=str,
        choices=L_LFM_GLOBAL, nargs="+", default=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc-inf", type=str,
        choices=L_LFM_LOCAL, nargs="+", default=L_LFM_LOCAL)

    d_a = ap.parse_args()

    return d_a

def create_true_config_folder_path(
        args: Dict[str, Any],
        h_true: float, tau_true: float,
        lfm_global_true: str, lfm_tc_true: str):
    return os.path.join(
        args.path_results,
        (f"{args.prefix}"
         f"lfm-g-true-{lfm_global_true}_lfm-t-true-{lfm_tc_true}_"
         f"h-true-{h_true}_tau-true-{tau_true}"
         "/"))

def create_inf_folder_name(
        lfm_global_inf: str, lfm_tc_inf: str):
    return (f"lfm-g-inf-{lfm_global_inf}_lfm-t-inf-{lfm_tc_inf}/")

def main():
    print("ELFI APS\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing',
                     num_processes=args.n_processes)

    np.random.seed(0)

    _n_combin = len(args.h_true) * len(args.tau_true)\
        * len(args.lfm_global_true) * len(args.lfm_tc_true)\
        * len(args.lfm_global_inf) * len(args.lfm_tc_inf)
    _run = 0
    print((
        f"Running for a total of "
        f"{_n_combin} combinations"))
    for h_true, tau_true, lfm_global_true, lfm_tc_true\
        in product(args.h_true, args.tau_true, args.lfm_global_true, args.lfm_tc_true):
        print((
            f"\tRunning true config h={h_true}, tau={tau_true}, "
            f"lfm_global={lfm_global_true}, lfm_tc={lfm_tc_true}..."))
        try:
            model_config = ModelConfig(
                N=N_NODES_SIM, f_m=F, m=M,
                realization=-1, # Will be ignored
                homophily=h_true, tau=tau_true,
                lfm_global=lfm_global_true, lfm_tc=lfm_tc_true)
        except ValueError as e:
            _run += len(args.lfm_global_inf) * len(args.lfm_tc_inf)
            print(f"\tSkipping combination: {e} ({_run} runs...)")
            continue

        print("\tSimulating observed graph...")
        graph_obs, t_edges_obs = elfi_patch(
            N=N_NODES_SIM, f_m=F, m=M,
            lfm_global=lfm_global_true, lfm_tc=lfm_tc_true,
            h=h_true, tau=tau_true, random_state=0
        )
        nodes_min = graph_obs.get_node_class(CLASS_ATTRIBUTE)

        m = max(2, compute_m(graph_empirical=graph_obs))
        f_m = np.mean(nodes_min)
        print((
            f"\tSimulated observed graph with {len(graph_obs)} nodes, "
            f"{len(t_edges_obs) // 2} edges, and "
            f"`f_m={f_m:.2f}`, `m={m}`"))

        for lfm_global_inf, lfm_tc_inf in product(args.lfm_global_inf, args.lfm_tc_inf):
            _run +=1
            print(f"\t\tRun {_run}/{_n_combin}")
            print((
                f"\t\tCreating simulator (`N={N_NODES_SIM}, m={m}, f_m={f_m:.2f}, "
                f"lfm_global={lfm_global_inf}, lfm_tc={lfm_tc_inf}`)..."))
            try:
                model_config = ModelConfig(
                    N=N_NODES_SIM, f_m=F, m=M,
                    realization=-1, # Will be ignored
                    homophily=-1, tau=-1, # Will be ignored
                    lfm_global=lfm_global_inf, lfm_tc=lfm_tc_inf)
            except ValueError as e:
                print(f"\tSkipping combination: {e}")
                continue

            simulator = create_elfi_simulator(
                model_config=model_config,
                observed=(graph_obs, t_edges_obs))

            # Define summary statistics
            summary_f = register_summary_stats(simulator)

            sampler = register_sampler(
            summary_f=summary_f)
            print("\t\tRunning sampling (this might take a while)...")
            # sample = sampler.sample(
                # N_SAMPLES, [0.7, 0.2, 0.05])
            sample = sampler.sample(
                N_SAMPLES, 2)

            file_posteriors = os.path.join(
                create_true_config_folder_path(
                    args=args,
                    h_true=h_true, tau_true=tau_true,
                    lfm_global_true=lfm_global_true, lfm_tc_true=lfm_tc_true),
                create_inf_folder_name(
                    lfm_global_inf=lfm_global_inf, lfm_tc_inf=lfm_tc_inf),
                "posteriors.npz")
            if not os.path.exists(os.path.dirname(file_posteriors)):
                os.makedirs(os.path.dirname(file_posteriors))
            print(f"\t\tSaving posteriors to `{file_posteriors}`...")
            np.savez(
                file=file_posteriors,
                h=sample.samples["h"],
                tau=sample.samples["tau"],
                discrepancies=sample.discrepancies,
                distance_weights=sample.adaptive_distance_w)

        print("\tComputing summary statistics for true graph:")
        s_observed = compute_observed_summary_stats(
            l_observed=[(graph_obs, t_edges_obs)],
            summary_f=summary_f)
        for k, v in s_observed.items():
            print(f"\t`{k}`: {v}")

        file_true_summary = os.path.join(
            create_true_config_folder_path(
                args=args,
                h_true=h_true, tau_true=tau_true,
                lfm_global_true=lfm_global_true, lfm_tc_true=lfm_tc_true),
            "summary.npz")
        if not os.path.exists(os.path.dirname(file_posteriors)):
            os.makedirs(os.path.dirname(file_posteriors))
        print(f"\tSaving summary statistics for true graph to `{file_true_summary}`...")
        np.savez(
            file=file_true_summary,
            **s_observed)

if __name__ == "__main__":
    main()
