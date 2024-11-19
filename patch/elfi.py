from typing import NamedTuple, Callable

from netin.models import PATCHModel, CompoundLFM
from netin.graphs import Graph
import numpy as np

from .statistics import compute_gini, compute_ei, compute_mann_whitney, compute_clustering_coefficient, compute_gini_maj, compute_gini_min

def elfi_patch(
        N: int, m: int, f_m: float,
        lfm_global: CompoundLFM, lfm_tc: CompoundLFM,
        random_state=None):
    return lambda h, tau: PATCHModel(
        N=N, f_m=f_m, m=m,
        tau=float(tau), h_M=float(h), h_m=float(h),
        lfm_global=lfm_global, lfm_tc=lfm_tc,
        random_state=random_state).simulate()

def elfi_gini(graph: Graph) -> float:
    return compute_gini(graph.degrees())

def elfi_ei(graph: Graph):
    return (compute_ei(graph) + 1) / 2

def compute_m(graph_empirical: Graph) -> int:
    n_nodes = len(graph_empirical)
    n_edges = graph_empirical.number_of_edges()

    return int((n_nodes - 1) - np.sqrt((n_nodes - 1) ** 2 - 2 * n_edges))

class ELFISummaryFunctions(NamedTuple):
    ei: Callable[[Graph], float] = elfi_ei
    gini: Callable[[Graph], float] = elfi_gini
    gini_min: Callable[[Graph], float] = compute_gini_maj
    gini_maj: Callable[[Graph], float] = compute_gini_min
    mann_whitney: Callable[[Graph], float] = compute_mann_whitney
    clustering_coefficient: Callable[[Graph], float] = compute_clustering_coefficient
