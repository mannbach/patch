from multiprocessing import Queue, Process
from queue import Empty
from argparse import ArgumentParser
from typing import Dict, Any, NamedTuple
import csv
import os
import json

import numpy as np

from patch.constants import\
    STOP_SIGNAL, PATH_GRAPHS, PATH_STATISTICS
from patch.io import read_graph_from_json
from patch.model_config import ModelConfig
from patch.statistics import compute_gini, compute_ei, compute_mann_whitney

def parse_args() -> Dict[str, Any]:
    """Parses the command line arguments.

    Returns
    -------
    Dict[str, Any]
        The parsed arguments.
    """
    ap = ArgumentParser("Aggregate Statistics")
    ap.add_argument("--path-graphs", "-pg",
                    default=PATH_GRAPHS, type=str)
    ap.add_argument("--path-results", "-pr",
                    default=PATH_STATISTICS, type=str)
    ap.add_argument("--n-processes", default=1, type=int)

    d_a = ap.parse_args()

    return d_a

class StatsResult(NamedTuple):
    """Provides a named tuple to store the results of the aggregate statistics.
    """
    model_config: ModelConfig
    gini: float
    ei: float
    mann_whitney: float
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
        return super().default(obj)

def work(queue_tasks: Queue, queue_results: Queue, folder_graphs: str):
    """Works on tasks from the queue_tasks and stores the results in the queue_results.
    For each task, the graph is loaded from the specified file, the aggregate statistics are computed and stored in the results queue.

    Parameters
    ----------
    queue_tasks : Queue
        The queue containing the tasks to be processed.
    queue_results : Queue
        The queue to store the results.
    folder_graphs : str
        The folder containing the graph files.
    """
    while not queue_tasks.empty():
        task = None
        try:
            task = queue_tasks.get(block=False)
        except Empty:
            return
        if task == STOP_SIGNAL:
            return
        i, file_graph = task

        print(f"Working on ({i}) {task}")

        # Generate the graph
        graph, data = read_graph_from_json(
            os.path.join(folder_graphs, file_graph))


        # Store minority nodes in a set
        model_config = ModelConfig.from_dict(data)

        # Compute the aggregate statistics
        stats = StatsResult(
            model_config=model_config,
            gini=compute_gini(graph),
            ei=compute_ei(graph),
            mann_whitney=compute_mann_whitney(graph)
        )

        # Put stats and JSON string into results queue
        queue_results.put(stats)

def main():
    """Creates all requested parameter combinations and processes them in parallel.
    """
    args = parse_args()

    # Multiprocess queues storing tasks and results
    queue_task = Queue()
    queue_results = Queue()

    # Create all parameter combinations
    n_combs = 0
    for task in enumerate(os.listdir(args.path_graphs)):
        queue_task.put(task)
        n_combs += 1

    # Add stop signals to the queue
    for _ in range(args.n_processes):
        queue_task.put(STOP_SIGNAL)

    # Start worker processes
    processes = []
    for _ in range(args.n_processes):
        p = Process(target=work, args=(queue_task, queue_results, args.path_graphs))
        p.start()
        processes.append(p)

    # Write incoming results to files
    print(f"Starting to write results to {args.path_results}")
    with open(args.path_results, 'w+', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        # Write CSV header
        csv_writer.writerow(StatsResult._fields)

        # Write until number of expected results reached
        while n_combs > 0:
            stats = queue_results.get()

            # Write JSON graph to file
            csv_writer.writerow(stats)

            n_combs -= 1

    print("Waiting for processes to join.")
    for p in processes:
        p.join()
    print("Done")

if __name__ == "__main__":
    main()
