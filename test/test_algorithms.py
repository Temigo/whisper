# -*- coding: utf-8 -*-

from algorithms import SZNode, SZEdge, SZGraph, SZ
import unittest

__author__ = 'temigo'


class SZTest(unittest.TestCase):
    def test_elementary(self):
        node1 = SZNode(id=1)
        node2 = SZNode(id=2)
        node3 = SZNode(id=3)
        edge1 = SZEdge(node1, node2)
        edge2 = SZEdge(node1, node3)
        graph = SZGraph([node1, node2, node3], [edge1, edge2])
        tree = graph.build_tree(node1)
        print(graph)

        s = SZ(graph)
        s.compute_rumor_centrality(tree)
        print(graph)