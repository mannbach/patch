"""This script contains functions to interact with the ELFI inference package.
"""
from typing import NamedTuple, Callable, List, Optional

import numpy as np
from netin.models import PATCHModel, CompoundLFM
from netin.graphs import Graph
import elfi

from .statistics import (
    compute_gini, compute_ei, compute_mann_whitney,
    compute_gini_maj, compute_gini_min, compute_average_ccf,
    compute_gini_comp)
from .model_config import ModelConfig
from .constants import N_NODES_SIM

def elfi_patch(
        N:int, f_m:float, m: int,
        lfm_global: CompoundLFM, lfm_tc: CompoundLFM,
        h: float, tau: float,
        random_state: np.random.RandomState) -> List[Graph]:
    """Simulate a network using the PATCHModel and return the graph and temporal edge list.

    Parameters
    ----------
    N : int
        The number of nodes in the network.
    f_m : float
        The fraction of minority nodes.
    m : int
        The number of links per node.
    lfm_global : CompoundLFM
        The global link formation model.
    lfm_tc : CompoundLFM
        The triadic closure link formation model.
    h : float
        The homophily parameter.
    tau : float
        The tau parameter for the model.
    random_state : np.random.RandomState
        The random state for reproducibility.

    Returns
    -------
    List[Graph]
        A list containing the simulated graph.
    """
    model = PATCHModel(
        n=int(N), f_m=float(f_m), m=int(m),
        tau=float(tau), h_MM=float(h), h_mm=float(h),
        lfm_global=CompoundLFM[lfm_global],
        lfm_tc=CompoundLFM[lfm_tc],
        random_state=random_state)

    graph = model.simulate()
    return [graph]

# Wrapper functions for ELFI summary statistics
def elfi_gini(res: List[Graph]) -> float:
    """Wrapper for `compute_gini` to be used in ELFI."""
    return np.mean([compute_gini(graph.degrees()) for graph in res])

def elfi_ei(res: List[Graph]):
    """Wrapper for `compute_ei` to be used in ELFI."""
    return np.mean([(compute_ei(graph) + 1) / 2 for graph in res])

def elfi_gini_maj(res: List[Graph]) -> float:
    """Wrapper for `compute_gini_maj` to be used in ELFI."""
    return np.mean([compute_gini_maj(graph) for graph in res])

def elfi_gini_min(res: List[Graph]) -> float:
    """Wrapper for `compute_gini_min` to be used in ELFI."""
    return np.mean([compute_gini_min(graph) for graph in res])

def elfi_gini_comp(res: List[Graph]) -> float:
    """Wrapper for `compute_gini_comp` to be used in ELFI."""
    return np.mean([compute_gini_comp(graph) for graph in res])

def elfi_mann_whitney(res: List[Graph]) -> float:
    """Wrapper for `compute_mann_whitney` to be used in ELFI."""
    return np.mean([compute_mann_whitney(graph) for graph in res])

def elfi_ccf(res: List[Graph]) -> np.ndarray:
    """Wrapper for `compute_average_ccf` to be used in ELFI."""
    return np.mean([compute_average_ccf(graph) for graph in res])

def compute_m(
        graph_empirical: Graph,
        n_nodes_sim: int = N_NODES_SIM) -> int:
    """Return the number of edges per node to reproduce the empirical average degree.

    Parameters
    ----------
    graph_empirical : Graph
        The empirical graph from which the average degree is computed.
    n_nodes_sim : int, optional
        The number of nodes in the simulated graph, by default N_NODES_SIM

    Returns
    -------
    int
        The number of edges per node to reproduce the empirical average degree.
    """
    n_nodes = len(graph_empirical)
    n_edges = graph_empirical.number_of_edges()
    return int(np.rint(
        (n_nodes_sim - (1/2))\
            - np.sqrt((n_nodes_sim - (1/2))**2 - (2 * n_edges / n_nodes) * n_nodes_sim)
    ))

def create_elfi_simulator(
        model_config: ModelConfig,
        params_constant: bool = False
        ) -> elfi.Simulator:
    """Create an ELFI simulator for the PATCH model.

    Parameters
    ----------
    model_config : ModelConfig
        The model configuration containing the parameters for the PATCH model.
    params_constant : bool, optional
        Whether to keep the parameters constant during optimization, by default False.
        This is useful for debugging or the predictive analysis.
        If `True`, the parameters are set to the values in `model_config` and not optimized.

    Returns
    -------
    elfi.Simulator
        The ELFI simulator for the PATCH model.
    """
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
        l_observations: Optional[List[Graph]] = None)\
            -> List[elfi.Summary]:
    """Register the summary statistics functions to the ELFI simulator.

    Parameters
    ----------
    simulator : elfi.Simulator
        The ELFI simulator to which the summary statistics functions are registered.
    l_observations : Optional[List[Graph]], optional
        The list of observed graphs, by default None.
        If provided, the summary statistics functions will compute the statistics
        based on these observations.

    Returns
    -------
    List[elfi.Summary]
        The list of registered summary statistics functions.
    """

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
    """ Register the sampler for the ELFI simulator.

    Parameters
    ----------
    summary_sim : List[elfi.Summary]
        The list of summary statistics functions to be used in the sampler.
    pool : Optional[elfi.ArrayPool], optional
        The output pool for the sampler, by default None.
        This is used to store the results of the simulation.

    Returns
    -------
    elfi.AdaptiveDistanceSMC
        The registered sampler for the ELFI simulator.
    """
    distance = elfi.AdaptiveDistance(*summary_sim)
    sampler = elfi.AdaptiveDistanceSMC(
        distance, pool=pool)
    return sampler

def create_pool(summary_f: List[elfi.Summary])\
        -> elfi.OutputPool:
    """Create an output pool for the ELFI simulator.

    Parameters
    ----------
    summary_f : List[elfi.Summary]
        The list of summary statistics functions to be used in the output pool.

    Returns
    -------
    elfi.OutputPool
        The output pool for the ELFI simulator.
    """
    return elfi.OutputPool([s.name for s in summary_f])

class ELFISummaryFunctions(NamedTuple):
    """The summary statistics functions to be used in the ELFI simulator.
    """
    ei: Callable[[Graph], float] = elfi_ei
    gini_min: Callable[[Graph], float] = elfi_gini_min
    gini_maj: Callable[[Graph], float] = elfi_gini_maj
    mann_whitney: Callable[[Graph], float] = elfi_mann_whitney
    ccf: Callable[[Graph], np.ndarray] = elfi_ccf
