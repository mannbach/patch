"""Scripts to compute aggregate inequality network statistics.
"""
from typing import Tuple, Dict, Set

from netin.utils.constants import CLASS_ATTRIBUTE
from netin.graphs import Graph, NodeVector
import numpy as np
import scipy as sc

from .temporal_edge_list import TemporalEdgeList

def compute_ei(net: Graph) -> float:
    """Compute the EI index of the network as a measure of network segregation.

    Parameters
    ----------
    net : Graph
        The simulated network.

    Returns
    -------
    Tuple[float, float]
        The EI-index where values close to 1 indicate a network in which nodes of one
        group prefer to connect to the other groups.
        Values close to -1 indicate segregation, as nodes prefer to connect their own group.
    """
    cnt_mM, cnt_mm, cnt_MM = 0, 0, 0
    nodes_min = net.get_node_class(CLASS_ATTRIBUTE).get_minority_mask()

    for u,v in net.edges():
        u_min, v_min = nodes_min[u], nodes_min[v]
        if u_min and v_min:
            cnt_mm += 1
        elif u_min != v_min: # XOR
            cnt_mM += 1
        elif not (u_min or v_min): # both maj
            cnt_MM += 1
    cnt_h = cnt_mm + cnt_MM
    return (cnt_mM - cnt_h) / (cnt_mM + cnt_h)

def compute_gini(degrees: NodeVector) -> float:
    """Computes the Gini coefficient of the degree distribution of the network.

    Parameters
    ----------
    net : nx.Graph
        The simulated network.

    Returns
    -------
    float
        The gini coefficient of the degree distribution.
        Values close to 0 indicate a more equal distribution of degrees.
        Values close to 1 indicate a more unequal distribution of degrees.
    """
    sorted_x = np.sort(degrees)
    n = len(degrees)
    cumx = np.cumsum(sorted_x, dtype=float)

    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def compute_gini_min(graph: Graph) -> float:
    degrees = graph.degrees()
    nodes_min = graph.get_node_class(CLASS_ATTRIBUTE)
    return compute_gini(degrees[nodes_min.get_minority_mask()])

def compute_gini_maj(graph: Graph) -> float:
    degrees = graph.degrees()
    nodes_min = graph.get_node_class(CLASS_ATTRIBUTE)
    return compute_gini(degrees[nodes_min.get_majority_mask()])

def compute_mann_whitney(net: Graph) -> float:
    """Computes the Mann-Whitney U test statistic for the degree distribution of the minority and majority groups.

    Parameters
    ----------
    net : Graph
        The simulated network.

    Returns
    -------
    float
        The Mann-Whitney U test statistic for the degree distribution of the minority
        and majority groups.
        Values close to 0.5 indicate that the degree distributions are similar.
        Values close to 1 indicate that the degree distribution of the minority group
        exceeds that of the majority group and values close to 0 indicate the opposite.
    """
    nodes_min = net.get_node_class(CLASS_ATTRIBUTE)
    degrees = net.degrees()

    k_min, k_maj = degrees[nodes_min.get_minority_mask()],\
        degrees[nodes_min.get_majority_mask()]

    return sc.stats.mannwhitneyu(k_min, k_maj).statistic / (len(k_min) * len(k_maj))

def _prepare_forward_neighbors(graph: Graph) -> Dict[int, Set[int]]:
    """Prepares a dictionary of forward neighbors for each node in the graph.

    Parameters
    ----------
    graph : Graph
        The simulated network.

    Returns
    -------
    Dict[int, Set[int]]
        A dictionary of forward neighbors for each node in the graph.
    """
    degrees = graph.degrees()
    forward = {}
    for u in graph.nodes():
        forward[u] = {v for v in graph.neighbors(u)\
            if (degrees[u] < degrees[v]) or (degrees[u] == degrees[v] and u < v)}
    return forward

def compute_ccf(graph: Graph, typed: bool = False) -> np.ndarray:
    degrees = graph.degrees()
    forward = _prepare_forward_neighbors(graph)
    nodes_min = graph.get_node_class(CLASS_ATTRIBUTE)

    t_count = np.zeros(4 if typed else 1)
    for u in graph.nodes():
        for v in forward[u]:
            for w in forward[u].intersection(forward[v]):
                t_count[np.sum(nodes_min[(u,v,w)]) if typed else 0] += 1

    # Count total number of connected triplets in the graph.
    total_triplets = np.zeros_like(t_count)
    for u in graph.nodes():
        k = degrees[u]
        if k >= 2:
            if typed:
                u_min = nodes_min[u]
                k_min = np.sum(nodes_min[forward[u]])
                k_maj = k - k_min

                total_triplets[u_min + 2] += k_min * (k_min - 1) / 2
                total_triplets[u_min + 1] += k_min * k_maj
                total_triplets[u_min] += k_maj * (k_maj - 1) / 2.
            else:
                total_triplets[0] += k * (k - 1) / 2

    if not np.any(total_triplets != 0):
        return np.zeros(1)

    # Global clustering coefficient:
    global_clustering = (3 * t_count) / total_triplets
    return global_clustering


def get_cdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the cumulative distribution function (CDF) of the data.

    Parameters
    ----------
    data : np.ndarray
        The data to compute the CDF for.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The x and y values of the CDF.
    """
    sorted_data = np.sort(data)
    yvals = np.arange(len(sorted_data)) / float(len(sorted_data))
    return sorted_data, yvals

def compute_group_ccf(res: Tuple[Graph, TemporalEdgeList]) -> np.ndarray:
    """Compute the clustering coefficient for each possible triangle combination based on the minority and majority groups.

    Parameters
    ----------
    graph : Graph
        The simulated network.

    Returns
    -------
    np.ndarray
        The clustering coefficients for each possible triangle combination.
        Returns a list of clustering coefficients ordered by the number of minority nodes in the triangle:
        0: mmm
        1: mmM
        2: mMm
        3: Mmm
        4: mMM
        5: MmM
        6: MMm
        7: MMM
    """
    graph, t_edges = res
    tri_cnts = np.zeros(8)
    nodes_min = graph.get_node_class(CLASS_ATTRIBUTE)

    for u, v in graph.edges():
        t_uv = t_edges[(u, v)]
        for w in graph.neighbors(u).intersection(graph.neighbors(v)):
            t_vw = t_edges[(v, w)]
            t_uw = t_edges[(u, w)]

            a,b,c = None, None, None
            if t_uv < t_vw < t_uw:
                a,b,c = u,v,w
            elif t_uv < t_uw < t_vw:
                a,b,c = v,u,w
            else:
                continue

            a_min, b_min, c_min = nodes_min[a], nodes_min[b], nodes_min[c]

            tri_cnts[
                0 if a_min and b_min and c_min else
                1 if a_min and b_min and not c_min else
                2 if a_min and not b_min and c_min else
                3 if not a_min and b_min and c_min else
                4 if a_min and not b_min and not c_min else
                5 if not a_min and b_min and not c_min else
                6 if not a_min and not b_min and c_min else
                7] += 1

    n_min = np.sum(nodes_min)
    n_maj = len(nodes_min) - n_min

    return tri_cnts / np.array([
        n_min * (n_min - 1) * (n_min - 2),
        n_min * (n_min - 1) * n_maj,
        n_min * (n_min - 1) * n_maj,
        n_min * (n_min - 1) * n_maj,
        n_min * (n_maj - 1) * n_maj,
        n_min * (n_maj - 1) * n_maj,
        n_min * (n_maj - 1) * n_maj,
        n_maj * (n_maj - 1) * (n_maj - 2)])

def compute_contour_lines(
    a_tau: np.ndarray, a_h: np.ndarray,
    percentiles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y = np.meshgrid(
        np.linspace(0, 1, 250),
        np.linspace(0, 1, 250))

    # Create kernel density estimate
    kde = sc.stats.gaussian_kde(
        np.vstack([a_tau, a_h]))

    # Evaluate KDE on grid
    Z = kde(np.vstack([X.ravel(), Y.ravel()]))
    Z = np.reshape(Z, X.shape)

    # Sort grid points by density in descending order
    sorted_idx = np.argsort(Z.ravel())[::-1]
    sorted_Z = Z.ravel()[sorted_idx]

    cumulative_Z = np.cumsum(sorted_Z) / np.sum(sorted_Z)

    thresholds = []
    for percentile in percentiles:
        # Find the index of the threshold value that contains the desired percentile
        threshold_idx = np.searchsorted(cumulative_Z, percentile)

        if threshold_idx < len(sorted_Z):
            thresholds.append(sorted_Z[threshold_idx])
        else:
            thresholds.append(sorted_Z[-1])

    return (
        X, Y, Z,
        np.array(thresholds)
    )
