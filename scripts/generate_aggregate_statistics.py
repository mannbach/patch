from multiprocessing import Queue, Process
from queue import Empty
from itertools import product
from argparse import ArgumentParser
from typing import Dict, Any, NamedTuple
import csv

from netin import PATCH, TCH, ERPATCH
from netin.utils.constants import\
    PATCH_MODEL_NAME, ERPATCH_MODEL_NAME, TCH_MODEL_NAME,\
    CLASS_ATTRIBUTE, MINORITY_VALUE

from patch_workshop.inequality_metrics import\
    compute_gini, compute_stoch_dom
from patch_workshop.homophily_metrics import compute_ei_index

STOP_SIGNAL = -1
PATH_FOLDER_GRAPHS = "./data/graphs/"
PATH_RESULTS = "./data/aggregate_statistics/aggregate_statistics.csv"
N = 5000
M = 2
MIN_FRACS = [0.3]
HOMOPHILY = [0.01, 0.2, 0.5, 0.8, 0.99]
TC = [0., 0.2, 0.5, 0.8, 1.]
REAL = 50
MODELS = [TCH_MODEL_NAME, PATCH_MODEL_NAME, ERPATCH_MODEL_NAME]

N_JOBS = 25

def parse_args() -> Dict[str, Any]:
    ap = ArgumentParser()
    ap.add_argument("--path-results", "-pr", default=PATH_RESULTS, type=str)
    ap.add_argument("-N", default=N, type=int)
    ap.add_argument("-m", default=M, type=int)
    ap.add_argument("-f", default=MIN_FRACS, nargs="+", type=float)
    ap.add_argument("-H", default=HOMOPHILY, nargs="+", type=float)
    ap.add_argument("-tc", default=TC, nargs="+", type=float)
    ap.add_argument("--realizations", "-r", default=REAL, type=int)
    ap.add_argument("--include-tcu", action="store_true", default=False)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--n-processes", default=1, type=int)

    d_a = ap.parse_args()

    d_a.tcu = [False, True] if d_a.include_tcu else [False]

    return d_a

class StatsResult(NamedTuple):
    f: float
    h: float
    tc: float
    tcu: bool
    r: int
    model_name: str
    gini: float
    ei: float
    stoch_dom: float

def work(queue_tasks: Queue, queue_results: Queue):
    while not queue_tasks.empty():
        task = None
        try:
            task = queue_tasks.get(block=False)
        except Empty:
            return
        if task == STOP_SIGNAL:
            break
        i, n_comb, (f, h, tc, tcu, r, model_name) = task

        print(f"Working on {task} ({i}/{n_comb})")
        Model = PATCH
        if model_name == TCH_MODEL_NAME:
            Model = TCH
        elif model_name == ERPATCH_MODEL_NAME:
            Model = ERPATCH

        graph = Model(n=N, f_m=f, k=M, h_mm=h, h_MM=h, tc=tc, tc_uniform=tcu)
        graph.generate()

        nodes_min = set(n for n, d in graph.nodes(data=True) if d[CLASS_ATTRIBUTE] == MINORITY_VALUE)

        queue_results.put(StatsResult(
            f=f, h=h, tc=tc, tcu=tcu,
            r=r, model_name=model_name,
            gini=compute_gini(graph),
            ei=compute_ei_index(graph, nodes_min)[2],
            stoch_dom=compute_stoch_dom(graph, nodes_min)
        ))

def main():
    args = parse_args()
    realizations = list(range(args.realizations))
    n_combs = args.realizations\
        * len(args.tc)\
        * len(args.H)\
        * len(args.f)\
        * len(args.models)\
        * len(args.tcu)

    queue_task = Queue()
    queue_results = Queue()

    for task in zip(
        range(n_combs),
        [n_combs for _ in range(n_combs)],
        product(args.f,
                args.H,
                args.tc,
                args.tcu,
                realizations,
                args.models)):
        queue_task.put(task)
    for _ in range(args.n_processes):
        queue_task.put(STOP_SIGNAL)

    processes = []
    for _ in range(args.n_processes):
        p = Process(target=work, args=(queue_task, queue_results))
        p.start()
        processes.append(p)

    print(f"Starting to write results to {args.path_results}")
    n_combs_cnt = 0
    with open(args.path_results, 'w', encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([
            "f", "h", "tc", "tcu", "r", "model_name",
            "gini", "ei", "stoch_dom"
        ])
        while n_combs_cnt < n_combs:
            csv_writer.writerow(queue_results.get())
            n_combs_cnt += 1

    print("Waiting for processes to join.")
    for p in processes:
        p.join()
    print("Done")

if __name__ == "__main__":
    main()
