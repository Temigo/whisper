# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:54:59 2016

@author: h
"""

# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - Shah and Zaman algorithm based on rumor centrality

# from graph import Node, Edge, Graph
import networkx as nx
from math import factorial


class AlgorithmSZ:
    def __init__(self):
        pass

    def run(self, graph, i_graph, v=None):
        """
        Run on a general graph
        :param graph:
        :param v: root of the spanning tree (v has to be infected !)
        :type graph: nx.Graph
        :type v: nx.Node
        :return:
        """
        # Spanning tree of infected graph

        # BFS on infected_nodes
        # FIXME source arbitrary ? iterate over sources ? because output changes according to chosen v
        # NB : output even changes when v is fixed
        tree = nx.bfs_tree(i_graph, source=v)

        # TODO return a list of equal probability rumor sources
        return self.algorithm_tree(tree, v)

    def algorithm_tree(self, tree, v=None):
        """
        Evaluates rumor source in a tree by computing rumor centrality for each node
        Exact estimation for regular trees, heuristic approach for general trees
        :param tree: a *regular* or general tree
        :type tree: nx.DiGraph
        :return:
        """
        if v is None:
            v = tree.nodes()[0]  # FIXME arbitrary choice ?

        r = self.compute_permitted_permutations(tree, v)
        # nx.set_node_attributes(tree, 'rumor_centrality', r)

        # Find the node with maximum rumor centrality
        max_rumor_centrality = 0
        source_estimation = None
        for node in tree:
            if r[node] > max_rumor_centrality:
                source_estimation = node
                max_rumor_centrality = r[node]
        return source_estimation

    @staticmethod
    def compute_permitted_permutations(origin_tree, v):
        """
        Computes R(v, G_N) (linear complexity)
        :param origin_tree:
        :param v:
        :type origin_tree: nx.DiGraph
        :type v: nx.Node
        :return:
        """
        # We need to traverse the tree in depth order : down-up and up-down
        tree = list(nx.dfs_postorder_nodes(origin_tree))
        t = {}
        p = {}
        r = {}

        # Down-up pass
        for u in tree:
            if origin_tree.neighbors(u) == []:  # u is a leaf
                t[u] = 1
                p[u] = 1
            else:
                t[u] = sum(t[child] for child in origin_tree.successors(u)) + 1
                p[u] = t[u]
                for child in origin_tree.successors(u):
                    p[u] *= p[child]

        # Up-down pass
        tree.reverse()
        N = origin_tree.number_of_nodes()  # or len(origin_tree)
        for u in tree:
            if u == v:
                r[v] = factorial(N)
                for child in origin_tree.successors(v):
                    r[v] /= p[child]
            else:
                r[u] = r[origin_tree.predecessors(u)[0]] * t[u] / (N - t[u])

        return r
