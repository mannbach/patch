"""Generates graphs for the given parameters and stores them as JSON.
"""
import os
from queue import Empty
from multiprocessing import Queue, Process
from itertools import product
from argparse import ArgumentParser
from typing import Dict, Any

from netin.models import PATCHModel

from patch.io import write_graph_to_json, create_file_name
from patch.model_config import ModelConfig
from patch.constants import (
    N, M, F, L_HOMOPHILY, L_TAU, N_REALIZATIONS,
    L_LFM_GLOBAL, L_LFM_LOCAL,
    PATH_GRAPHS, STOP_SIGNAL
)

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("Graph Generator")
    ap.add_argument("--path", "-p", default=PATH_GRAPHS, type=str)
    ap.add_argument("-N", default=N, type=int)
    ap.add_argument("-m", default=M, type=int)
    ap.add_argument("-f", default=[F], nargs="+", type=float)
    ap.add_argument("-H", default=L_HOMOPHILY, nargs="+", type=float)
    ap.add_argument("-tau", default=L_TAU, nargs="+", type=float)
    ap.add_argument("--realizations", "-r",
                    default=N_REALIZATIONS, type=int)
    ap.add_argument("-lfm-g",
                    nargs="+",
                    default=L_LFM_GLOBAL, type=str, choices=L_LFM_GLOBAL)
    ap.add_argument("-lfm-t",
                    nargs="+",
                    default=L_LFM_LOCAL, type=str, choices=L_LFM_LOCAL)
    ap.add_argument("--n-processes", default=1, type=int)

    d_a = ap.parse_args()

    assert os.path.isdir(d_a.path)

    return d_a

def work(queue_tasks: Queue, path: str):
    """Takes tasks from the queue_tasks and processes them.
    For each task, a graph is generated and and stored as JSON.

    Parameters
    ----------
    queue_tasks : Queue
        The queue containing the tasks to be processed.
    """
    while not queue_tasks.empty():
        task = None
        try:
            task = queue_tasks.get(block=False)
        except Empty:
            return
        if task == STOP_SIGNAL:
            return
        i, model_config = task

        print(f"Working on (seed={i}) {task}")

        # Generate the graph
        graph = PATCHModel(
            **model_config.to_dict(patch_model=True),
            seed=i).simulate()

        write_graph_to_json(
            **model_config.to_dict(
                patch_model=True, stringify=True),
            path=os.path.join(
                path,
                create_file_name(model_config=model_config)),
            graph=graph
        )

def main():
    """Fills task queue with all possible combinations of parameters and
    starts the processes to work on them.
    """
    args = parse_args()
    realizations = list(range(args.realizations))
    n_combs = args.realizations\
        * len(args.tau)\
        * len(args.H)\
        * len(args.f)\
        * len(args.lfm_g)\
        * len(args.lfm_t)
    print(f"Number of combinations: {n_combs}")
    queue = Queue()
    for i, (f, h, tau, lfm_g, lfm_l, real)  in enumerate(
        product(args.f,
                args.H,
                args.tau,
                args.lfm_g,
                args.lfm_t,
                realizations)):
        try:
            config = ModelConfig(
            N=args.N,
            m=args.m,
            f_m=f,
            homophily=h,
            tau=tau,
            lfm_global=lfm_g,
            lfm_tc=lfm_l,
            realization=real)
            queue.put((i, config))
        except ValueError as e:
            print(f"Skipping {i} due to {e}")

    for _ in range(args.n_processes):
        queue.put(STOP_SIGNAL)

    processes = []
    for _ in range(args.n_processes):
        p = Process(target=work, args=(queue, args.path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done")

if __name__ == "__main__":
    main()
