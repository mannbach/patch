from typing import List, Tuple

import pytest
import numpy as np
from netin.graphs import Graph

from patch.statistics import compute_ccf

# Language: Python

# A create_fake_graph to simulate minimal Graph functionality for tests.
def create_fake_graph(nodes: List[int], edges: List[Tuple[int, int]]) -> Graph:
    graph = Graph()
    for node in nodes:
        graph.add_node(node)
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    return graph

def test_ccf_empty():
    # Graph with no nodes should throw a ValueError.
    graph = create_fake_graph([], [])

    with pytest.raises(ValueError):
        compute_ccf(graph)

def test_ccf_triangle():
    # Triangle graph: 3 nodes fully connected.
    # Expected global clustering: each node forms one triplet -> total_triplets = 3 and t_count = 1, so CCF=(3*1)/3=1.0.
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (0, 2)]
    graph = create_fake_graph(nodes, edges)
    result = compute_ccf(graph)
    expected = 1.0
    assert np.isclose(result, expected, atol=1e-6)

def test_ccf_line3():
    # Line graph: nodes: 0-1-2, only one triplet at the middle node but no triangle (t_count = 0).
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    graph = create_fake_graph(nodes, edges)
    result = compute_ccf(graph)
    # total_triplets: only node 1 contributes: (2*1)/2 = 1, t_count = 0 => CCF = 0.0.
    expected = 0.0
    assert np.isclose(result, expected, atol=1e-6)
