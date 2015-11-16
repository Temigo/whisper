# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - Fioriti and Chinnici algorithm

# from graph import Node, Edge, Graph
import networkx as nx
import numpy as np
from math import fabs


class AlgorithmFC:
    def __init__(self):
        pass

    @staticmethod
    def run(graph):
        """
        :param graph:
        :type graph: nx.Graph
        :return:
        """
        dynamical_age = {}
        lambda_m = np.amax(nx.linalg.spectrum.adjacency_spectrum(graph))
        max_node = None

        for node in graph:
            if graph.node[node]['infected']:
                # Compute Dynamical Age (DA) of node
                g2 = graph.copy()
                g2.remove_node(node)
                lambda_m2 = np.amax(nx.linalg.spectrum.adjacency_spectrum(g2))

                dynamical_age[node] = fabs(lambda_m - lambda_m2) / lambda_m
                if max_node is None:
                    max_node = node
                elif dynamical_age[node] > dynamical_age[max_node]:
                    max_node = node

        # TODO sort and return k most likely sources
        return max_node
