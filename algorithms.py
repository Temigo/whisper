# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph

__author__ = 'temigo'

from graph import Graph, Node, Edge


class PTVNode(Node):
    def __init__(self):
        super().__init__()
        self.is_observer = False
        self.is_source = False
        self.observations = []

    def add_observation(self, neighbor, time):
        self.observations.append((neighbor, time))


class PTVEdge(Edge):
    def __init__(self, source, destination):
        super().__init__(source, destination)
        self.random_delay_propagation = None


class PTVGraph(Graph):
    def __init__(self, nodes, edges):
        super().__init__(nodes, edges)
        # List observers
        self.observers = []
        for node in nodes:
            if node.is_observer:
                self.observers.append(node)


class PTV:
    """
    Algorithm PTV (for Pinto-Thiran-Vetterli - not an official name !)
    Reference :
     - Locating the Source of Diffusion in Large-Scale Networks.
    Pedro C. Pinto, Patrick Thiran, and Martin Vetterli. 2012 (PHYSICAL REVIEW LETTERS)
    """
    def __init__(self, graph):
        self.graph = graph


