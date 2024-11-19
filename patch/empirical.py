from typing import Tuple
import os
from itertools import product

from netin.graphs import BinaryClassNodeVector
from netin.utils.constants import MINORITY_VALUE, MAJORITY_VALUE, MINORITY_LABEL, MAJORITY_LABEL, CLASS_ATTRIBUTE
import pandas as pd
from netin.graphs import Graph
import networkx as nx

GENDER_UNKNOWN = 0
GENDER_FEMALE = 1
GENDER_MALE = 2

def read_graph(folder: str)\
    -> Graph:
    df_authorships = pd.read_csv(
        os.path.join(
            folder, "authorships.csv"), index_col=0)
    df_publications = pd.read_csv(
        os.path.join(folder, "publications.csv"),
        index_col="id_publication")
    df_author_name = pd.read_csv(
        os.path.join(folder, "author_names.csv"),
        index_col="id_author_name")
    df_authors = pd.read_csv(
        os.path.join(folder, "authors.csv"),
        index_col="id_author")

    df_authors = df_authors[
        df_authors["id_gender_nq"] != GENDER_UNKNOWN]
    df_authors[CLASS_ATTRIBUTE] = df_authors["id_gender_nq"]\
        .map(lambda g: MINORITY_VALUE if g == GENDER_FEMALE else MAJORITY_VALUE)
    df_authors["index_new"] = range(len(df_authors))

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
        .sort_values(by=["timestamp"], ascending=True)\
        .groupby("id_publication")

    graph = Graph()
    nodes_min = BinaryClassNodeVector(
        N=len(df_authors),
        class_labels=[MAJORITY_LABEL, MINORITY_LABEL])

    for author, minority in df_authors[['index_new', CLASS_ATTRIBUTE]].itertuples(index=False):
        graph.add_node(author)
        nodes_min[author] = minority

    for _, df_auth_pub in df_edges:
        for u, v in product(df_auth_pub["id_author"], repeat=2):
            if u != v:
                u_new = df_authors.loc[u, "index_new"]
                v_new = df_authors.loc[v, "index_new"]
                if not graph.has_edge(u_new, v_new):
                    graph.add_edge(u_new, v_new)

    graph.set_node_class(CLASS_ATTRIBUTE, nodes_min)

    return graph
