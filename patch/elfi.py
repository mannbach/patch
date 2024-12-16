from typing import NamedTuple, Callable, Tuple

from netin.models import PATCHModel, CompoundLFM
from netin.graphs import Graph
from netin.utils.event_handling import Event
import numpy as np

from .temporal_edge_list import TemporalEdgeList
from .statistics import compute_gini, compute_ei, compute_mann_whitney, compute_group_ccf, compute_gini_maj, compute_gini_min

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
    return graph, t_edges

def elfi_gini(res: Tuple[Graph, TemporalEdgeList]) -> float:
    graph = res[0]
    return compute_gini(graph.degrees())

def elfi_ei(res: Tuple[Graph, TemporalEdgeList]):
    graph = res[0]
    return (compute_ei(graph) + 1) / 2

def elfi_gini_maj(res: Tuple[Graph, TemporalEdgeList]) -> float:
    graph = res[0]
    return compute_gini_maj(graph)

def elfi_gini_min(res: Tuple[Graph, TemporalEdgeList]) -> float:
    graph = res[0]
    return compute_gini_min(graph)

def elfi_mann_whitney(res: Tuple[Graph, TemporalEdgeList]) -> float:
    graph = res[0]
    return compute_mann_whitney(graph)

def compute_m(graph_empirical: Graph) -> int:
    n_nodes = len(graph_empirical)
    n_edges = graph_empirical.number_of_edges()
    return int((((2*n_nodes) - 1) / 2)\
               - (np.sqrt(((2*n_nodes) - 1) ** 2 - (8 * n_edges)) / 2))

def d_cosine(*simulated, observed):
    return 1 - np.dot(simulated, observed) / (np.linalg.norm(simulated) * np.linalg.norm(observed))

class ELFISummaryFunctions(NamedTuple):
    ei: Callable[[Graph, TemporalEdgeList], float] = elfi_ei
    gini: Callable[[Graph, TemporalEdgeList], float] = elfi_gini
    gini_min: Callable[[Graph, TemporalEdgeList], float] = elfi_gini_maj
    gini_maj: Callable[[Graph, TemporalEdgeList], float] = elfi_gini_min
    mann_whitney: Callable[[Graph, TemporalEdgeList], float] = elfi_mann_whitney
    # group_ccf: Callable[[Graph, TemporalEdgeList], np.ndarray] = compute_group_ccf
