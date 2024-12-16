from typing import Tuple
import os
from itertools import product

from netin.graphs import BinaryClassNodeVector
from netin.graphs import Graph
from netin.utils.constants import MINORITY_VALUE, MAJORITY_VALUE, MINORITY_LABEL, MAJORITY_LABEL, CLASS_ATTRIBUTE
import pandas as pd
import networkx as nx

from .temporal_edge_list import TemporalEdgeList

GENDER_UNKNOWN = 0
GENDER_FEMALE = 1
GENDER_MALE = 2

def read_graph(folder: str, decade: int, duration: int = 10)\
    -> Tuple[Graph, TemporalEdgeList]:
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

    df_authors = df_authors[df_authors["disambiguated"]]
    df_authors = df_authors[
        df_authors["id_gender_nq"] != GENDER_UNKNOWN]
    df_authors[CLASS_ATTRIBUTE] = df_authors["id_gender_nq"]\
        .map(lambda g: MINORITY_VALUE if g == GENDER_FEMALE else MAJORITY_VALUE)

    df_publications = df_publications[
        (df_publications["timestamp"].dt.year >= decade) & (df_publications["timestamp"].dt.year <= (decade + duration))]

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
               how="inner")\
        .sort_values(by=["timestamp"], ascending=True)


    # Keep only authors who have published after decade + duration
    authors_active = df_edges.groupby("id_author")["timestamp"].max().dt.year >= (decade + duration)

    df_authors = df_authors[df_authors.index.isin(authors_active.index)]
    df_authors = df_authors[authors_active]

    # df_authors = df_authors[
    #     df_edges.groupby("id_author")["timestamp"].max().dt.year >= (decade + duration)]
    map_auth_old_new = {}
    map_auth_new_group = {}
    id_auth = 0

    df_edges = df_edges[df_edges["id_author"].isin(df_authors.index)]

    gb_edges = df_edges.groupby("id_publication")

    graph = Graph()
    edge_times = {}

    # for author, minority in df_authors[['index_new', CLASS_ATTRIBUTE]].itertuples(index=False):
    #     graph.add_node(author)
    #     nodes_min[author] = minority

    time = -1
    time_old = None
    for _, df_auth_pub in gb_edges:
        if len(df_auth_pub) == 1:
            continue
        tmp_curr = df_auth_pub["timestamp"].iloc[0]
        for u, v in product(df_auth_pub["id_author"], repeat=2):
            for x in u, v:
                if not x in map_auth_old_new:
                    map_auth_old_new[x] = id_auth
                    map_auth_new_group[id_auth] = df_authors.loc[x, CLASS_ATTRIBUTE]
                    graph.add_node(id_auth)
                    id_auth += 1

            u_new = map_auth_old_new[u]
            v_new = map_auth_old_new[v]

            if u_new != v_new:
                if not graph.has_edge(u_new, v_new):
                    graph.add_edge(u_new, v_new)
                    if time_old != tmp_curr:
                        time += 1
                        time_old = tmp_curr
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
