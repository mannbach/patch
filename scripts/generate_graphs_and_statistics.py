from multiprocessing import Queue, Process
from itertools import product
from queue import Empty
from argparse import ArgumentParser
from typing import Dict, Any, NamedTuple
import csv
import os
import json

import numpy as np
import networkx as nx

from netin.utils.constants import\
    CLASS_ATTRIBUTE, MINORITY_VALUE

from patch_workshop.inequality_metrics import\
    compute_gini, compute_stoch_dom
from patch_workshop.homophily_metrics import compute_ei_index
from patch_workshop.utils import create_file_name, create_graph
from patch_workshop.constants import LFM_RANDOM, LFM_HOMOPHILY, LFM_PAH

# Constants
STOP_SIGNAL = -1 # Signal to stop the worker processes

# Default parameters
# Can be changed by runnig supplying the arguments in the command line
# (see parse_args)
PATH_RESULTS = "./data/aggregate_statistics/aggregate_statistics.csv"
N_JOBS = 10 # Number of processes
N = 5000 # Number of nodes
M = 2 # Number of edges per node
MIN_FRACS = [0.3] # Fraction of minority nodes
HOMOPHILY = [0.01, 0.2, 0.5, 0.8, 0.99] # Homophily
TC = [0., 0.2, 0.5, 0.8, 1.] # Triadic closure probability
REAL = 50 # Number of realizations
LFMS = [LFM_RANDOM, LFM_HOMOPHILY, LFM_PAH] # Local and global link formation mechanisms

def parse_args() -> Dict[str, Any]:
    ap = ArgumentParser()
    ap.add_argument("--path-results", "-pr",
                    default=PATH_RESULTS, type=str)
    ap.add_argument(
        "--path-graphs", "-pg",
        default=None, type=str, required=False)
    ap.add_argument("-N", default=N, type=int)
    ap.add_argument("-m", default=M, type=int)
    ap.add_argument("-f", default=MIN_FRACS, nargs="+", type=float)
    ap.add_argument("-H", default=HOMOPHILY, nargs="+", type=float)
    ap.add_argument("-tc", default=TC, nargs="+", type=float)
    ap.add_argument("--lfm-global", "-lfmg", default=LFMS, nargs="+", type=str)
    ap.add_argument("--lfm-local", "-lfml", default=LFMS, nargs="+", type=str)
    ap.add_argument("--realizations", "-r", default=REAL, type=int)
    ap.add_argument("--n-processes", default=1, type=int)

    d_a = ap.parse_args()

    return d_a

class StatsResult(NamedTuple):
    """Provides a named tuple to store the results of the aggregate statistics.
    """
    f: float
    h: float
    tc: float
    lfm_global: str
    lfm_local: str
    r: int
    gini: float
    ei: float
    stoch_dom: float
    json_data: str = None

class NpEncoder(json.JSONEncoder):
    """Encoder for numpy types to be used in json.dumps.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def work(queue_tasks: Queue, queue_results: Queue):
    """Takes tasks from the queue_tasks and processes them.
    For each task, a graph is generated and the aggregate statistics are computed.
    The results are stored in the queue_results.

    Parameters
    ----------
    queue_tasks : Queue
        The queue containing the tasks to be processed.
    queue_results : Queue
        The queue where the results are stored.
    """
    while not queue_tasks.empty():
        task = None
        try:
            task = queue_tasks.get(block=False)
        except Empty:
            return
        if task == STOP_SIGNAL:
            return
        i, path_graphs, (n, m, f, h, tc, lfm_g, lfm_l, r) = task

        print(f"Working on {task} ({i})")

        # Generate the graph
        graph = create_graph(
            N=n, m=m, f_m=f,
            h=h, tc=tc,
            lfm_global=lfm_g, lfm_local=lfm_l
        )

        # If graphs should be stored, convert the graph to a json string
        json_str = None
        if path_graphs is not None:
            graph.graph["h"] = h
            graph.graph["tc"] = tc
            graph.graph["lfm_l"] = lfm_l
            graph.graph["lfm_g"] = lfm_g
            graph.graph["realization"] = r
            json_str = json.dumps(
                nx.node_link_data(graph),
                cls=NpEncoder
            )

        # Store minority nodes in a set
        nodes_min = set(
            n for n, d in graph.nodes(data=True) if d[CLASS_ATTRIBUTE] == MINORITY_VALUE)

        # Compute the aggregate statistics
        stats = StatsResult(
            f=f, h=h, tc=tc,
            lfm_global=lfm_g, lfm_local=lfm_l,
            r=r,
            gini=compute_gini(graph),
            ei=compute_ei_index(graph, nodes_min)[2],
            stoch_dom=compute_stoch_dom(graph, nodes_min)
        )

        # Put stats and JSON string into results queue
        queue_results.put((stats, json_str))

def main():
    """Creates all requested parameter combinations and processes them in parallel.
    """
    args = parse_args()
    # Compute number of combinations
    n_combs = args.realizations\
        * len(args.tc)\
        * len(args.H)\
        * len(args.f)\
        * len(args.lfm_global)\
        * len(args.lfm_local)

    # Multiprocess queues storing tasks and results
    queue_task = Queue()
    queue_results = Queue()

    # Create all parameter combinations
    realizations = list(range(args.realizations))
    for task in zip(
        range(n_combs),
        [args.path_graphs for _ in range(n_combs)],
        product(
            [args.N],
            [args.m],
            args.f,
            args.H,
            args.tc,
            args.lfm_global,
            args.lfm_local,
            realizations)):
        lfm_local, lfm_global = task[2][-2], task[2][-3]

        # Skip invalid link formation mechanism combinations
        if lfm_local not in (LFM_RANDOM, lfm_global):
            n_combs -= 1
            print((f"Skipping {task} because lfm_local={lfm_local} is not in ({LFM_RANDOM}, "
                   f"lfm_global={lfm_global})\nNew n_combs={n_combs}"))
            continue

        # Add task to queue
        queue_task.put(task)

    # Add stop signals to the queue
    for _ in range(args.n_processes):
        queue_task.put(STOP_SIGNAL)

    # Start worker processes
    processes = []
    for _ in range(args.n_processes):
        p = Process(target=work, args=(queue_task, queue_results))
        p.start()
        processes.append(p)

    # Write incoming results to files
    print(f"Starting to write results to {args.path_results}")
    n_combs_cnt = 0
    with open(args.path_results, 'w+', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        # Write CSV header
        csv_writer.writerow([
            "f", "h", "tc", "tcu", "r", "model_name",
            "gini", "ei", "stoch_dom"
        ])

        # Write until number of expected results reached
        while n_combs_cnt < n_combs:
            stats, json_graph = queue_results.get()

            if stats.lfm_local not in (LFM_RANDOM, stats.lfm_global):
                n_combs_cnt += 1
                continue

            # Write JSON graph to file
            if json_graph is not None:
                file_path = os.path.join(args.path_graphs, create_file_name(
                    N=args.N, m=args.m, minority_fraction=stats.f,
                    homophily=stats.h, triadic_closure=stats.tc,
                    lfm_local=stats.lfm_local, lfm_global=stats.lfm_global,
                    realization=stats.r, file_ending=".json"
                ))
                print(f"Writing graph to {file_path}")
                with open(file_path, 'w+', encoding="utf-8") as f:
                    f.writelines(json_graph)

            csv_writer.writerow(stats)
            n_combs_cnt += 1

    print("Waiting for processes to join.")
    for p in processes:
        p.join()
    print("Done")

if __name__ == "__main__":
    main()
