"""Tests for the computation of the average clustering coefficient (CCF) in graphs.
"""
from typing import List, Tuple

import pytest
import numpy as np
from netin.graphs import Graph

from patch.statistics import compute_average_ccf

# A create_fake_graph to simulate minimal Graph functionality for tests.
def create_fake_graph(nodes: List[int], edges: List[Tuple[int, int]]) -> Graph:
    """Creates a fake graph for testing purposes.

    Parameters
    ----------
    nodes : List[int]
        Nodes to add to the graph.
    edges : List[Tuple[int, int]]
        Edges to add to the graph.

    Returns
    -------
    Graph
        A Graph object representing the fake graph.
    """
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
        compute_average_ccf(graph)

def test_ccf_triangle():
    # Triangle graph: 3 nodes fully connected.
    # Expected global clustering: each node forms one
    # triplet -> total_triplets = 3 and t_count = 1, so CCF=(3*1)/3=1.0.
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2), (0, 2)]
    graph = create_fake_graph(nodes, edges)
    result_average = compute_average_ccf(graph)
    expected = 1.0
    assert np.isclose(result_average, expected, atol=1e-6)

def test_ccf_line3():
    # Line graph: nodes: 0-1-2, only one triplet at the middle node but no triangle (t_count = 0).
    nodes = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    graph = create_fake_graph(nodes, edges)
    result_average = compute_average_ccf(graph)
    # total_triplets: only node 1 contributes: (2*1)/2 = 1, t_count = 0 => CCF = 0.0.
    expected = 0.0
    print(f"CCF for line graph: {result_average}")
    assert np.isclose(result_average, expected, atol=1e-6)

def test_ccf_tri_plus_one():
    # Test a triangle with one additional node connected to one of the triangle nodes.
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (0, 2), (2, 3)]
    graph = create_fake_graph(nodes, edges)
    result_average = compute_average_ccf(graph)

    # Expected average CCF:
    # Nodes 0 and 1 have only neighbors in triangle: contribute local clustering of 1.0
    # Node 3 has no closed triangle: contributes 0.0
    # Node 2 has one closed and two open triangles: contributes 1/3
    # Average should be 1/4 * (1 + 1 + 0 + 1/3) = 7/12
    expected_average = 7 / 12

    assert np.isclose(result_average, expected_average, atol=1e-6)
