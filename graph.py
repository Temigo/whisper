# -*- coding: utf-8 -*-
# Graph classes


class Node:
    def __init__(self):
        self.neighbors = []

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)


class Edge:  # A voir : si cette classe est nécessaire
    def __init__(self, source, destination, oriented=False):
        self.source = source
        self.destination = destination

        self.source.add_neighbor(self.destination)
        if not oriented:
            self.destination.add_neighbor(self.source)

    def __repr__(self):
        return "[%s -> %s]" % (self.source.__repr__(), self.destination.__repr__())


class Graph:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges
