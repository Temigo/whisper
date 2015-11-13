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

    def run(self, graph):
        """
        :param graph:
        :type graph: nx.Graph
        :return:
        """
        # FIXME where is the infection involved ?!
        dynamical_age = {}
        lambda_m = np.amax(nx.linalg.spectrum.adjacency_spectrum(graph))
        max_node = None

        for node in graph:
            g2 = graph.copy()
            g2.remove_node(node)
            lambda_m2 = np.amax(nx.linalg.spectrum.adjacency_spectrum(g2))
            dynamical_age[node] = fabs(lambda_m - lambda_m2) / lambda_m
            if max_node==None:
                max_node = node
            elif dynamical_age[node] > dynamical_age[max_node]:
                max_node = node

        # TODO sort
        return max_node
