# -*- coding: utf-8 -*-
# Base graph model

__author__ = 'temigo'


class Node:
    def __init__(self):
        self.id = None
        self.neighbors = None


class Observer(Node):
    pass


class Edge:
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.weight = None

    @property
    def get_source(self):
        return self.source

    @property
    def get_destination(self):
        return self.destination


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
        self.nodes_count = len(nodes)
        self.edges_count = len(edges)
