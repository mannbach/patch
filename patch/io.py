from itertools import product
import json
import os
from typing import Any, Dict, Tuple, Union, List, Generator

import networkx as nx
from netin.models import PATCHModel
from netin.graphs import Graph

from .model_config import ModelConfig

def read_graph_from_json(path: str) -> Tuple[PATCHModel, Dict[str, Union[int, float]]]:
    """Read a graph from a json file.
    Returns tuple of the graph and a dict that holds the remaining information
    stored in the file (e.g. minority nodes, generation parameters, ...).

    Args:
        path (str): Path to the json file.

    Returns:
        Union[PATCHModel, Dict[str, Union[int, float]]]: Tuple containing the graph
            and dict that contains additional information.
    """
    graph = None
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        graph = nx.from_edgelist(data["edge_list"])
    del data["edge_list"]
    return graph, data

def write_graph_to_json(
    path: str,
    graph: Graph,
    **kwargs):
    """Write a graph's edge list to a json file.
    The edge list will be stored under the key "edge_list".
    Additional data (specified in **kwargs) will be stored next to "edge_list" at the top level.

    Args:
        path (str): Path to write the json to.
        graph (Graph): Graph to store.
        **kwargs (Any): Additional data that will be stored at the top level of the json file.
    """
    data = {}
    for key, val in kwargs.items():
        data[key] = val
    data["edge_list"] = [(int(u), int(v)) for (u,v) in graph.edges]
    with open(path, 'w', encoding="utf-8") as file:
        file.writelines(json.dumps(data, indent=2))

def create_file_name(
        model_config: ModelConfig,
        prefix: str = "",
        suffix: str = "",
        file_ending: str = ".json") -> str:
    """Creates and returns a filename string describing the configuration given by the parameters.
    This is useful for centralizing I/O operations under a unified terminology.

    Args:
        model_config (ModelConfig): The model configuration to create the filename
        prefix (str, optional): Set if a prefix should be added to the string. Defaults to "".
        suffix (str, optional): Set if a suffix should be added to the string. Defaults to "".
        file_ending (str, optional): The file ending. Defaults to ".json".

    Returns:
        str: The filename string.
    """
    return (
        f"{prefix}"
        f"N-{model_config.N}_m-{model_config.m}_"
        f"f-{model_config.f_m}_"
        f"h-{model_config.homophily}_"
        f"tau-{model_config.tau}_"
        f"lfm-g-{model_config.lfm_global}_"
        f"lfm-t-{model_config.lfm_tc}_"
        f"r-{model_config.realization}{suffix}{file_ending}"
    )

def gen_nets_from_file(
        path: str,
        N:int,
        m: int,
        f: float,
        l_homophily: List[float],
        l_tau: List[float],
        l_lfm_g: List[str],
        l_lfm_t: List[str],
        n_realizations: int,
        **kwargs) -> Generator[None, Tuple[Graph, Dict[str, Any]], None]:
    """Generates networks and metadata tuples for all possible
    combinations of the specified parameters.
    Networks are read from json files in the specified folder path.

    Parameters
    ----------
    path : str
        Path to the folder containing the json files.
    N : int
        Number of nodes.
    m : int
        Number of edges to attach from a new node to existing nodes.
    f : float
        Fraction of minority nodes.
    l_homophily : List[float]
        List of homophily values.
    l_tau : List[float]
        List of tau values.
    l_lfm_g : List[str]
        List of global LFM values.
    l_lfm_t : List[str]
        List of triadic closure LFM values.
    n_realizations : int
        Number of realizations.

    Yields
    ------
    Generator[None, Tuple[Graph, Dict[str, Any]], None]
        Generator that yields tuples of networks and metadata dictionaries.
    """
    for h, tau, lfm_g, lfm_t, r in\
        product(l_homophily, l_tau, l_lfm_g, l_lfm_t, range(n_realizations)):
        net, info =\
            read_graph_from_json(os.path.join(path, create_file_name(
                ModelConfig(
                    N=N, m=m,
                    f_m=f,
                    homophily=h,
                    tau=tau,
                    lfm_global=lfm_g,
                    lfm_tc=lfm_t,
                    realization=r)
                **kwargs)))
        yield net, info
