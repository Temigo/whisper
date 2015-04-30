# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - SZ : Shah and Zaman algorithm based on rumor centrality

__author__ = 'temigo'

from graph import Graph, Node, Edge


def fact(n):
    if n == 1:
        return 1
    else:
        return fact(n-1) * n


class SZNode(Node):
    def __init__(self, id=None, infected=False):
        super().__init__(id=id)
        self.infected = infected

        self.parent = None
        self.children = []
        self.p = None
        self.t = None
        self.r = None
        self.depth = None

    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.depth = self.depth + 1

    def __repr__(self):
        if self.parent is not None:
            return "<%s SZNode/Parent: %s/R: %s>" % (self.id, self.parent.id, self.r)
        else:
            return "<%s SZNode/Parent: %s/R: %s>" % (self.id, self.parent, self.r)


class SZEdge(Edge):
    def __repr__(self):
        return "§" + self.source.__repr__() + " --> " + self.destination.__repr__() + "§"


class SZGraph(Graph):
    def __init__(self, nodes, edges):
        super().__init__(nodes, edges)
        self.infected_nodes = []
        for node in nodes:
            if node.infected:
                self.infected_nodes.append(node)

    @staticmethod
    def build_tree(root):
        root.depth = 0
        pile = [root]
        visited_nodes = []
        while pile:
            node = pile.pop()
            visited_nodes.append(node)
            for child in node.neighbors:
                if child not in visited_nodes:
                    node.add_child(child)
                    pile.append(child)
        return visited_nodes  # list of nodes ordered by depth

    def __repr__(self):
        return "SZGRAPH \n %s \n %s" % (self.nodes, self.edges)


class SZ:
    """
    Algorithm of Shah and Zaman
    Rumor centrality message-passing algorithm
    """
    def __init__(self, graph):
        self.graph = graph

    @staticmethod
    def compute_rumor_centrality(tree):
        """
        Rumor centrality message-passing algorithm
        For now we assume the graph is a regular tree
        :param tree: list of nodes ordered by depth
        :return:
        """
        N = len(tree)
        # Bottom-up
        tree.reverse()
        for node in tree:  # ordered by depth
            if not node.children:  # leaf
                node.t = 1
                node.p = 1
            else:
                if node.depth > 0:  # not the root
                    node.t = sum(child.t for child in node.children) + 1
                    node.p = node.t
                    for child in node.children:
                        node.p *= child.p
        tree.reverse()
        # Top-down
        for node in tree:
            if node.depth == 0:
                node.r = fact(N-1)
                for child in node.children:
                    node.r /= child.p
            else:
                node.r = node.parent.r * node.t / (N - node.t)

    def run(self):
        # For now we assume the graph is a regular tree
        pass