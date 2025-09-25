"""Performs predictive analysis using ELFI based on previously inferred posteriors.
"""
from argparse import ArgumentParser
from typing import Any, Dict
import os
from itertools import product
from multiprocessing import Queue, Process
from collections import defaultdict

import elfi
import numpy as np
from netin.models import PATCHModel

from patch.constants import (
    PATH_INFERENCE,
    PATH_INFERENCE_PREDICTIVE,
    N_SAMPLES_PREDICTIVE, N_NODES_SIM,
    L_LFM_GLOBAL, L_LFM_LOCAL,
    APS, DBLP, APS_CIT)
from patch.elfi import (
    ELFISummaryFunctions)
from patch.model_config import ModelConfig

STOP_SIGNAL = -1

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("ELFI predictive analysis")
    ap.add_argument("--source", "-s", type=str, choices=[APS, DBLP, APS_CIT])
    ap.add_argument("--path-source", "-ps", type=str, default=PATH_INFERENCE)
    ap.add_argument("--path-results", "-pr",
                    default=PATH_INFERENCE_PREDICTIVE, type=str)
    ap.add_argument("--decades", type=int, nargs="+")
    ap.add_argument("--n-processes", default=1, type=int)
    ap.add_argument("--n-samples", default=N_SAMPLES_PREDICTIVE, type=int)

    ap.add_argument("-lfm-global", type=str, choices=L_LFM_GLOBAL, nargs="+", default=L_LFM_GLOBAL)
    ap.add_argument("-lfm-tc", type=str, choices=L_LFM_LOCAL, nargs="+", default=L_LFM_LOCAL)

    ap.add_argument("--prefix", type=str, default="")

    d_a = ap.parse_args()

    return d_a

def create_folder_name(
        path_base: str, source: str,
        lfm_global: str, lfm_tc: str,
        decade: int,
        prefix: str = ""):
    """Creates the folder name for storing predictive results.

    Parameters
    ----------
    path_base : str
        Base path for the results.
    source : str
        Source dataset.
    lfm_global : str
        Global link formation mechanism model.
    lfm_tc : str
        Local link formation mechanism model.
    decade : int
        Decade for the data.
    prefix : str, optional
        Prefix for the folder name, by default ""

    Returns
    -------
    str
        The folder name for storing predictive results.
    """
    return os.path.join(
        path_base,
        source + "/",
        f"{prefix}lfm-g-{lfm_global}_lfm-t-{lfm_tc}_d-{decade}/")

def _work(
        q_task: Queue, q_result: Queue
):
    while True:
        task = q_task.get()
        if task == STOP_SIGNAL:
            break

        i, decade, model_config = task
        patch = PATCHModel(
            **model_config.to_dict(patch_model=True),
            seed=i
        )
        graph = patch.simulate()

        q_result.put((i, decade, model_config, {
            k: f([(graph, None)])\
                for k, f in ELFISummaryFunctions()._asdict().items()
        }))

def main():
    """Performs predictive analysis using ELFI based on previously inferred posteriors.
    """
    print("ELFI predictive analysis\nParsing args...")
    args = parse_args()

    print(f"Setting `n_processes` to {args.n_processes}")
    elfi.set_client('multiprocessing',
                     num_processes=args.n_processes)

    queue_task = Queue()
    queue_result = Queue()

    for decade, lfm_global, lfm_tc in product(
            args.decades, args.lfm_global, args.lfm_tc):
        file_posteriors = os.path.join(
                create_folder_name(
                    path_base=args.path_source,
                    source=args.source,
                    prefix=args.prefix,
                    lfm_global=lfm_global, lfm_tc=lfm_tc,
                    decade=decade),
                "posteriors.npz")
        print((f"\nReading posterior for ({lfm_global},{lfm_tc}) "
               f"`{decade}` from {file_posteriors}"))

        a_posteriors = np.load(file_posteriors, allow_pickle=True)

        for h_inf, tau_inf in zip(
                a_posteriors["h"], a_posteriors["tau"]):
            model_config = ModelConfig(
                N=N_NODES_SIM,
                f_m=float(a_posteriors["f_m"]),
                m=int(a_posteriors["m"]),
                realization=-1,
                homophily=float(h_inf), tau=float(tau_inf),
                lfm_global=lfm_global, lfm_tc=lfm_tc)
            for i in range(args.n_samples):
                queue_task.put((i, decade, model_config))
    n_tasks = queue_task.qsize()
    print(f"Filled task queue with {n_tasks} tasks.")

    for i in range(args.n_processes):
        queue_task.put(STOP_SIGNAL)

    print("Starting worker processes...")
    l_workers = [
        Process(
            target=_work,
            args=(queue_task, queue_result))\
                for _ in range(args.n_processes)]
    for worker in l_workers:
        worker.start()

    print("Workers started, waiting for results...")
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    while n_tasks > 0:
        i, decade, model_config, summaries = queue_result.get()
        for summary_name, summary_value in summaries.items():
            results\
                [(decade, model_config.lfm_global, model_config.lfm_tc)]\
                [(model_config.tau, model_config.homophily)]\
                [summary_name] += summary_value
        n_tasks -= 1

    print("All tasks completed, stopping workers...")
    for worker in l_workers:
        worker.join()

    print("Workers stopped, processing results...")
    for (decade, lfm_global, lfm_tc), summaries in results.items():
        folder_name = create_folder_name(
            path_base=args.path_results,
            lfm_global=lfm_global.value, lfm_tc=lfm_tc.value,
            prefix=args.prefix,
            source=args.source,
            decade=decade)
        if not os.path.exists(folder_name):
            print(f"Creating folder `{folder_name}`...")
            os.makedirs(folder_name)
        print(f"Writing results to `{folder_name}`...")

        # Taking the summary statistic averages for each posterior sample
        results_config = {}
        for summ_name in ELFISummaryFunctions()\
                ._asdict():
            results_posterior = results\
                [(decade, lfm_global, lfm_tc)]\
                .values()
            results_config[summ_name] = [
                vals[summ_name] / args.n_samples\
                    for vals in results_posterior]

        file_summary = os.path.join(folder_name, "predictive.npz")
        np.savez(
            file=file_summary,
            **results_config)
        print(f"Saved posteriors to `{file_summary}`...")

if __name__ == "__main__":
    main()
