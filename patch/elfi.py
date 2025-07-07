from typing import NamedTuple, Callable, Tuple, List, Optional

from netin.models import PATCHModel, CompoundLFM
from netin.graphs import Graph
from netin.utils.event_handling import Event
import numpy as np
import elfi

from .temporal_edge_list import TemporalEdgeList
from .statistics import compute_gini, compute_ei, compute_mann_whitney, compute_gini_maj, compute_gini_min, compute_ccf
from .model_config import ModelConfig

def elfi_patch(
        N:int, f_m:float, m: int,
        lfm_global: CompoundLFM, lfm_tc: CompoundLFM,
        h: float, tau: float,
        random_state: np.random.RandomState) -> Tuple[Graph, TemporalEdgeList]:
    time = 0
    t_edges = {}

    def link_add_handler(source, target):
        nonlocal time, t_edges
        t_edges[source, target] = time
        t_edges[target, source] = time
        time += 1

    model = PATCHModel(
        N=int(N), f_m=float(f_m), m=int(m),
        tau=float(tau), h_M=float(h), h_m=float(h),
        lfm_global=CompoundLFM[lfm_global], lfm_tc=CompoundLFM[lfm_tc],
        random_state=random_state)

    model.register_event_handler(
        event=Event.LINK_ADD_BEFORE,
        function=link_add_handler
    )

    graph = model.simulate()
    return [(graph, t_edges)]

def elfi_gini(res: List[Tuple[Graph, TemporalEdgeList]]) -> float:
    return np.mean([compute_gini(graph.degrees()) for graph, _ in res])

def elfi_ei(res: List[Tuple[Graph, TemporalEdgeList]]):
    return np.mean([(compute_ei(graph) + 1) / 2 for graph, _ in res])

def elfi_gini_maj(res: List[Tuple[Graph, TemporalEdgeList]]) -> float:
    return np.mean([compute_gini_maj(graph) for graph, _ in res])

def elfi_gini_min(res: List[Tuple[Graph, TemporalEdgeList]]) -> float:
    return np.mean([compute_gini_min(graph) for graph, _ in res])

def elfi_mann_whitney(res: List[Tuple[Graph, TemporalEdgeList]]) -> float:
    return np.mean([compute_mann_whitney(graph) for graph, _ in res])

def elfi_ccf(res: List[Tuple[Graph, TemporalEdgeList]]) -> np.ndarray:
    return np.mean([compute_ccf(graph)[0] for graph, _ in res])

def compute_m(graph_empirical: Graph) -> int:
    n_nodes = len(graph_empirical)
    n_edges = graph_empirical.number_of_edges()
    return int((((2*n_nodes) - 1) / 2)\
               - (np.sqrt(((2*n_nodes) - 1) ** 2 - (8 * n_edges)) / 2))

def d_cosine(*simulated, observed):
    return 1 - np.dot(simulated, observed) / (np.linalg.norm(simulated) * np.linalg.norm(observed))

def create_elfi_simulator(
        model_config: ModelConfig,
        params_constant: bool = False
        ) -> elfi.Simulator:
    model = elfi.ElfiModel()

    h_prior = elfi.Prior('uniform', 0, 1, model=model, name="h")\
        if not params_constant else\
            elfi.Constant(model_config.homophily, model=model, name="h")
    tau_prior = elfi.Prior('uniform', 0, 1, model=model, name="tau")\
        if not params_constant else\
            elfi.Constant(model_config.tau, model=model, name="tau")

    simulator = elfi.Simulator(
        elfi.tools.vectorize(
            elfi_patch, # Simulator function
            constants=(0, 1, 2, 3, 4) if not params_constant else\
                (0, 1, 2, 3, 4, 5, 6),
            dtype=False), # Non-array dtype of simulation
        model_config.N, model_config.f_m, model_config.m,
        model_config.lfm_global.value, model_config.lfm_tc.value,
        h_prior, tau_prior,
        name="simulator", # For later reference
        )

    return simulator

def register_summary_stats_functions(
        simulator: elfi.Simulator,
        l_observations: Optional[List[Tuple[Graph, TemporalEdgeList]]] = None)\
            -> List[elfi.Summary]:

    # Define summary statistics
    summary_f = [
        elfi.Summary(
            elfi.tools.vectorize(f), simulator,
            name=k,
            observed=f(l_observations) if l_observations is not None else None)
        for k, f in ELFISummaryFunctions()._asdict().items()
    ]
    return summary_f

def register_sampler(summary_sim: List[elfi.Summary],
                     pool: Optional[elfi.ArrayPool] = None)\
        -> elfi.AdaptiveDistanceSMC:
    distance = elfi.AdaptiveDistance(*summary_sim)
    sampler = elfi.AdaptiveDistanceSMC(
        distance, pool=pool)
    return sampler

def create_pool(summary_f: List[elfi.Summary]) -> elfi.OutputPool:
    return elfi.OutputPool([s.name for s in summary_f])

def _mean(data: np.ndarray, **kwargs):
    return np.mean(data, axis=1, **kwargs)

class ELFISummaryFunctions(NamedTuple):
    ei: Callable[[Graph, TemporalEdgeList], float] = elfi_ei
    gini: Callable[[Graph, TemporalEdgeList], float] = elfi_gini
    gini_min: Callable[[Graph, TemporalEdgeList], float] = elfi_gini_maj
    gini_maj: Callable[[Graph, TemporalEdgeList], float] = elfi_gini_min
    mann_whitney: Callable[[Graph, TemporalEdgeList], float] = elfi_mann_whitney
    ccf: Callable[[Graph, TemporalEdgeList], np.ndarray] = elfi_ccf
