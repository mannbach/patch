"""Scripts to compute aggregate inequality network statistics.
"""
from typing import Tuple

from netin.utils.constants import CLASS_ATTRIBUTE
from netin.graphs import Graph, NodeVector

import numpy as np
import scipy as sc

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
    nodes_min = net.get_node_class(CLASS_ATTRIBUTE)

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

    k_min, k_maj = degrees[nodes_min], degrees[np.invert(nodes_min)]

    return sc.stats.mannwhitneyu(k_min, k_maj).statistic / (len(k_min) * len(k_maj))
