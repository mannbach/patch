from itertools import product
import json
import os
from typing import Tuple, List, Generator

import numpy as np
from netin.graphs import Graph, BinaryClassNodeVector
from netin.utils.constants import CLASS_ATTRIBUTE

from .model_config import ModelConfig

def read_graph_from_json(path: str) -> Tuple[ModelConfig, Graph]:
    """Reads graphs and model config from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file.

    Returns
    -------
    Tuple[ModelConfig, Graph]
        The model config and read Graph.
    """
    graph = None
    with open(path, 'r', encoding="utf-8") as file:
        data = json.load(file)
        model_config = ModelConfig.from_dict(data)
        graph = Graph()
        for node in range(model_config.N):
            graph.add_node(node)
        for u, v in data["edge_list"]:
            graph.add_edge(u, v)
        node_ids_min = np.asarray(data[CLASS_ATTRIBUTE])
        # Convert IDs to boolean mask
        node_ids_mask = np.isin(np.arange(model_config.N), node_ids_min, assume_unique=True)
        graph.set_node_class(
            CLASS_ATTRIBUTE,
            BinaryClassNodeVector.from_ndarray(node_ids_mask))
    return model_config, graph

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
    node_ids = np.arange(len(graph))
    data[CLASS_ATTRIBUTE] = node_ids[graph.get_node_class(CLASS_ATTRIBUTE)\
                                     .get_minority_mask()].tolist()
    data["edge_list"] = [(int(u), int(v)) for (u,v) in graph.edges()]
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
        f"lfm-g-{model_config.lfm_global.value}_"
        f"lfm-t-{model_config.lfm_tc.value}_"
        f"r-{model_config.realization}{suffix}{file_ending}"
    )

def create_net_subfolder_name(
    N:int , m: int, f: float
) -> str:
    """Creates and returns a subfolder name string describing the
    configuration given by the parameters.

    Parameters
    ----------
    model_config : ModelConfig
        The model configuration to create the subfolder name.

    Returns
    -------
    str
        The subfolder name string.
    """
    return (
        f"N-{N}_m-{m}_f-{f}"
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
    **kwargs) -> Generator[None, Tuple[ModelConfig, Graph], None]:
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
    Generator[None, Tuple[ModelConfig, Graph], None]
        Generator that yields tuples of networks and model configuration.
    """
    for h, tau, lfm_g, lfm_t, r in\
        product(l_homophily, l_tau, l_lfm_g, l_lfm_t, range(n_realizations)):
        try:
            info, net =\
                read_graph_from_json(os.path.join(path, create_file_name(
                    ModelConfig(
                        N=N, m=m,
                        f_m=f,
                        homophily=h,
                        tau=tau,
                        lfm_global=lfm_g,
                        lfm_tc=lfm_t,
                        realization=r),
                    **kwargs)))
            yield info, net
        except ValueError as e:
            print((f"Error reading combination: {h}, {tau}, "
                   f"{lfm_g}, {lfm_t}, {r}. Message:\n{e}\nSkipping"))
