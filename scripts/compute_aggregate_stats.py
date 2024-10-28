from multiprocessing import Queue, Process
from queue import Empty
from argparse import ArgumentParser
from typing import Dict, Any, List
import csv
import os
import json
import dataclasses

import numpy as np
from netin.utils.constants import CLASS_ATTRIBUTE

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

@dataclasses.dataclass
class StatsResult:
    """Provides a named tuple to store the results of the aggregate statistics.
    """
    model_config: ModelConfig
    gini: float
    ei: float
    mann_whitney: float
    gini_min: float
    gini_maj: float
    json_data: str = None

    _CSV_FIELDS_STATS = ("gini", "ei", "mann_whitney")

    @staticmethod
    def get_csv_fields() -> List[str]:
        return tuple(field.name for field in dataclasses.fields(ModelConfig)) + StatsResult._CSV_FIELDS_STATS

    def get_csv_values(self) -> List[Any]:
        l_vals = []

        d_config = self.model_config.to_dict(stringify=True)
        for field in dataclasses.fields(ModelConfig):
            l_vals.append(d_config[field.name])

        for field in StatsResult._CSV_FIELDS_STATS:
            l_vals.append(getattr(self, field))

        return l_vals

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
        model_config, graph = read_graph_from_json(
            os.path.join(folder_graphs, file_graph))

        degrees = graph.degrees()
        nodes_min = graph.get_node_class(CLASS_ATTRIBUTE)

        # Compute the aggregate statistics
        stats = StatsResult(
            model_config=model_config,
            gini=compute_gini(degrees),
            ei=compute_ei(graph),
            mann_whitney=compute_mann_whitney(graph),
            gini_min=compute_gini(degrees[nodes_min.get_minority_mask()]),
            gini_maj=compute_gini(degrees[nodes_min.get_majority_mask()])
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
    path_file = os.path.join(args.path_results, "aggregate_statistics.csv")
    print(f"Starting to write results to {path_file}")
    with open(path_file, 'w+', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        # Write CSV header
        csv_writer.writerow(StatsResult.get_csv_fields())

        # Write until number of expected results reached
        while n_combs > 0:
            stats = queue_results.get()

            # Write JSON graph to file
            csv_writer.writerow(stats.get_csv_values())

            n_combs -= 1

    print("Waiting for processes to join.")
    for p in processes:
        p.join()
    print("Done")

if __name__ == "__main__":
    main()
