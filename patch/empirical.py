"""This script provides functions to handle the empirical data used in the inference analysis.
"""
from typing import Tuple, Optional
import os
from itertools import combinations

from netin.graphs import BinaryClassNodeVector
from netin.graphs import Graph
from netin.utils.constants import (
    MINORITY_VALUE, MAJORITY_VALUE, MINORITY_LABEL,
    MAJORITY_LABEL, CLASS_ATTRIBUTE)
import pandas as pd

from .temporal_edge_list import TemporalEdgeList
from .constants import APS, DBLP, APS_CIT, PATH_APS, PATH_DBLP

GENDER_UNKNOWN = 0
GENDER_FEMALE = 1
GENDER_MALE = 2

GENDER_UNKNOWN_DBLP = '-'
GENDER_FEMALE_DBLP = 'gf'

def read_graph(
        source: str, decade: int,
        duration: int = 10,
        folder: Optional[str] = None)\
            -> Tuple[Graph, TemporalEdgeList]:
    """Read a graph for a given source and decade.

    Parameters
    ----------
    source : str
        The source of the graph (e.g., "aps", "dblp").
    decade : int
        The decade for which to read the graph.
    duration : int, optional
        The duration in years for which to read the graph, by default 10.
    folder : str, optional
        The folder where the data is stored, by default None.

    Returns
    -------
    Tuple[Graph, TemporalEdgeList]
        The graph and the temporal edge list.
    """
    if source == APS:
        return read_graph_aps(
            folder=folder or PATH_APS,
            decade=decade,
            duration=duration)
    if source == DBLP:
        return read_graph_dblp(
            folder=folder or PATH_DBLP,
            decade=decade,
            duration=duration)
    if source == APS_CIT:
        return read_graph_aps_cit(
            folder=folder or PATH_APS,
            decade=decade,
            duration=duration)
    raise ValueError(f"Unknown source: {source}")

def read_graph_dblp(folder: str, decade: int, duration: int = 10)\
    -> Tuple[Graph, TemporalEdgeList]:
    """Read a graph from the DBLP dataset.

    Returns
    -------
    Tuple[Graph, TemporalEdgeList]
        The graph and the temporal edge list.
    """
    df_authors = pd.read_csv(
        os.path.join(folder, "ent.author"),
        sep=r"\s+",
        names=['author_id', 'author_name', 'gender'])\
            .set_index('author_id')
    df_edges = pd.read_csv(
        os.path.join(folder, "out.dblp_coauthor"),
        skiprows=1,
        sep=r"\s+",
        names=[
            'author_id1', 'author_id2', 'weight', 'timestamp'])

    df_edges = df_edges\
        .merge(df_authors, left_on="author_id1", right_index=True, suffixes=("", "_1"))\
        .merge(df_authors, left_on="author_id2", right_index=True, suffixes=("", "_2"))\
        .rename(columns={
            "gender": "gender_1",
        })

    df_edges["timestamp"] = pd.to_datetime(df_edges["timestamp"], unit="s")

    # Filter out authors with unknown genders
    df_edges = df_edges[
        (df_edges["gender_1"] != GENDER_UNKNOWN_DBLP)\
            & (df_edges["gender_2"] != GENDER_UNKNOWN_DBLP)
    ]

    # Keep only authors who have published after decade + duration
    authors_active_1 = df_edges\
        .groupby("author_id1")["timestamp"].max().dt.year >= (decade + duration)
    authors_active_2 = df_edges\
        .groupby("author_id2")["timestamp"].max().dt.year >= (decade + duration)
    authors_active = authors_active_1 | authors_active_2
    authors_active = authors_active[authors_active].index

    df_edges = df_edges[
        df_edges["author_id1"].isin(authors_active)\
            & df_edges["author_id2"].isin(authors_active)]

    df_edges = df_edges[
        (df_edges["timestamp"].dt.year >= decade)\
            & (df_edges["timestamp"].dt.year < (decade + duration))]

    authors_active = pd.concat([
        df_edges["author_id1"],
        df_edges["author_id2"]],
        axis=0).unique()

    df_authors = df_authors[df_authors.index.isin(authors_active)]
    df_authors[CLASS_ATTRIBUTE] = df_authors["gender"]\
        .map(lambda g: MINORITY_VALUE if g == GENDER_FEMALE_DBLP else MAJORITY_VALUE)

    map_auth_old_new = {}
    map_auth_new_group = {}
    id_auth = 0
    graph = Graph()

    for x in authors_active:
        if not x in map_auth_old_new:
            map_auth_old_new[x] = id_auth
            map_auth_new_group[id_auth] = df_authors.loc[x, CLASS_ATTRIBUTE]
            graph.add_node(id_auth)
            id_auth += 1

    df_edges = df_edges\
        .sort_values(by=["timestamp"],
                    ascending=True)
    time = -1
    time_last = None
    edge_times = {}
    for _, row in df_edges.iterrows():
        time_curr = row["timestamp"]
        if time_curr != time_last:
            if time_last is not None:
                assert time_last <= time_curr,\
                    f"Assertion failed: time_old ({time_last}) > tmp_curr ({time_curr})"
            time += 1
            time_last = time_curr

        u = map_auth_old_new[row["author_id1"]]
        v = map_auth_old_new[row["author_id2"]]
        if u != v:
            if not graph.has_edge(u, v):
                graph.add_edge(u, v)
                edge_times[(u, v)] = time
                edge_times[(v, u)] = time

    nodes_min = BinaryClassNodeVector(
        N=len(graph),
        class_labels=[MAJORITY_LABEL, MINORITY_LABEL])
    for node, minority in map_auth_new_group.items():
        nodes_min[node] = minority

    graph.set_node_class(
        CLASS_ATTRIBUTE,
        nodes_min)

    return graph, edge_times

def _read_aps_data(folder: str, include_cit: bool = False)\
    -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    df_authorships = pd.read_csv(
        os.path.join(
            folder, "authorships.csv"), index_col=0)
    df_publications = pd.read_csv(
        os.path.join(folder, "publications.csv"),
        index_col="id_publication",
        parse_dates=["timestamp"])
    df_author_name = pd.read_csv(
        os.path.join(folder, "author_names.csv"),
        index_col="id_author_name")
    df_authors = pd.read_csv(
        os.path.join(folder, "authors.csv"),
        index_col="id_author")

    df_citations = None
    if include_cit:
        df_citations = pd.read_csv(
            os.path.join(folder, "citations.csv"),
            index_col=0)
    return df_authorships, df_publications, df_author_name, df_authors, df_citations

def read_graph_aps(folder: str, decade: int, duration: int = 10)\
    -> Tuple[Graph, TemporalEdgeList]:
    df_authorships, df_publications, df_author_name, df_authors, _ = _read_aps_data(
        folder=folder)

    df_edges = df_authorships\
        .merge(df_publications,
               left_on="id_publication",
               right_index=True)\
        .merge(df_author_name,
               left_on="id_author_name",
               right_index=True)\
        .merge(df_authors,
               left_on="id_author",
               right_index=True,
               how="inner")

    # Filter out authors without disambiguation or unknown gender
    df_edges = df_edges[df_edges["disambiguated"]]
    df_edges = df_edges[
        df_edges["id_gender_nq"] != GENDER_UNKNOWN]

    # Keep only authors who have published after decade + duration
    authors_active = df_edges.groupby("id_author")["timestamp"].max().dt.year >= (decade + duration)
    authors_active = authors_active[authors_active].index
    df_edges = df_edges[df_edges["id_author"].isin(authors_active)]

    # Filter out papers outside of decade
    # This has to happen after filtering out authors because they rely on the papers
    df_edges = df_edges[
        (df_edges["timestamp"].dt.year >= decade)\
            & (df_edges["timestamp"].dt.year < (decade + duration))]

    # Add minority attribute
    df_authors = df_authors[df_authors.index.isin(authors_active)]
    df_authors[CLASS_ATTRIBUTE] = df_authors["id_gender_nq"]\
        .map(lambda g: MINORITY_VALUE if g == GENDER_FEMALE else MAJORITY_VALUE)

    map_auth_old_new = {}
    map_auth_new_group = {}
    id_auth = 0

    # Sort by timestamp and publication id
    df_edges = df_edges\
        .sort_values(by=["timestamp"],
                    ascending=True)
    gb_edges = df_edges.groupby("id_publication")

    graph = Graph()
    edge_times = {}

    for x in df_edges["id_author"].unique():
        if not x in map_auth_old_new:
            map_auth_old_new[x] = id_auth
            map_auth_new_group[id_auth] = df_authors.loc[x, CLASS_ATTRIBUTE]
            graph.add_node(id_auth)
            id_auth += 1

    time = -1
    time_last = None
    for id_publication in df_edges["id_publication"].unique():
        df_pub = gb_edges.get_group(id_publication)
        time_curr = df_pub["timestamp"].iloc[0]

        if time_curr != time_last:
            if time_last is not None:
                assert time_last <= time_curr,\
                    f"Assertion failed: time_old ({time_last}) > tmp_curr ({time_curr})"
            time += 1
            time_last = time_curr

        for u, v in combinations(df_pub["id_author"], 2):
            u_new = map_auth_old_new[u]
            v_new = map_auth_old_new[v]

            if u_new != v_new:
                if not graph.has_edge(u_new, v_new):
                    graph.add_edge(u_new, v_new)
                    edge_times[(u_new, v_new)] = time
                    edge_times[(v_new, u_new)] = time

    nodes_min = BinaryClassNodeVector(
        N=len(graph),
        class_labels=[MAJORITY_LABEL, MINORITY_LABEL])
    for node, minority in map_auth_new_group.items():
        nodes_min[node] = minority

    graph.set_node_class(
        CLASS_ATTRIBUTE,
        nodes_min)

    return graph, edge_times

def read_graph_aps_cit(
        folder: str, decade: int, duration: int = 10)\
    -> Tuple[Graph, TemporalEdgeList]:
    df_authorships, df_pub, df_name, df_authors, df_cit = _read_aps_data(
        folder=folder, include_cit=True)

    df_auth_first = pd.merge(
            df_authorships\
                .groupby("id_publication")\
                ["id_author_name"]\
                .first(),
            df_name,
            left_on="id_author_name",
            right_index=True)\
        .merge(df_authors,
               left_on="id_author",
               right_index=True)\
        .merge(df_pub,
               left_index=True,
               right_index=True)
    df_auth_first = df_auth_first[
        df_auth_first["disambiguated"]\
            & (df_auth_first["id_gender_nq"] != GENDER_UNKNOWN)\
            & (df_auth_first["timestamp"].dt.year < (decade + duration))]
    df_auth_first[CLASS_ATTRIBUTE] = df_auth_first["id_gender_nq"]\
        .map(lambda g: MINORITY_VALUE if g == GENDER_FEMALE else MAJORITY_VALUE)

    df_cit = df_cit\
        .merge(df_auth_first,
               left_on="id_publication_citing",
               right_index=True,
               # Inner join
               how="inner")\
        .merge(df_auth_first,
               left_on="id_publication_cited",
               right_index=True,
               how="inner",
               suffixes=("_citing", "_cited"))\
        .sort_values(
                by=["timestamp_citing"],
                ascending=True)

    graph = Graph()
    edge_times = {}

    map_auth_old_new = {}
    map_auth_new_group = {}

    id_auth = 0
    for x in set(df_cit["id_publication_citing"]).union(df_cit["id_publication_cited"]):
        if not x in map_auth_old_new:
            map_auth_old_new[x] = id_auth
            map_auth_new_group[id_auth] = df_auth_first.loc[x, CLASS_ATTRIBUTE]
            graph.add_node(id_auth)
            id_auth += 1

    time_last = None
    time = -1

    gb_cit = df_cit\
        .groupby("id_publication_citing")
    for id_pub_citing in df_cit['id_publication_citing'].unique():
        df_pub = gb_cit.get_group(id_pub_citing)
        time_curr = df_pub["timestamp_citing"].iloc[0]
        if time_curr != time_last:
            if time_last is not None:
                assert time_last <= time_curr,\
                    f"Assertion failed: time_old ({time_last}) > tmp_curr ({time_curr})"
            time += 1
            time_last = time_curr

        for id_pub_cited in df_pub["id_publication_cited"]:
            if id_pub_citing == id_pub_cited:
                continue
            u = map_auth_old_new[id_pub_citing]
            v = map_auth_old_new[id_pub_cited]
            if not graph.has_edge(u, v):
                graph.add_edge(u, v)
                edge_times[(u, v)] = time
                edge_times[(v, u)] = time

    nodes_min = BinaryClassNodeVector(
        N=len(graph),
        class_labels=[MAJORITY_LABEL, MINORITY_LABEL])
    for node, minority in map_auth_new_group.items():
        nodes_min[node] = minority

    graph.set_node_class(
        CLASS_ATTRIBUTE,
        nodes_min)

    return graph, edge_times
