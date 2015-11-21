# -*- coding: utf-8 -*-
# Tests of algorithm Pinto

from algorithm_pinto import AlgorithmPinto
import networkx as nx
import unittest


class TestPinto(unittest.TestCase):
    # Méthode appelée avant chaque test
    def setUp(self):
        pass

    # Méthode appelée après chaque test
    def tearDown(self):
        pass

    def test_elementary(self):
        """
            1
          /  \
         2   3
        :return:
        """
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3])
        g.add_edges_from([(1, 2), (1, 3)])  # FIXME do you need to add also (2, 1) and (3, 1) ?

        g.node[1]['infected'] = True
        g.node[2]['infected'] = True
        g.node[3]['infected'] = True

        g.node[2]['time'] = 2.
        g.node[3]['time'] = 2.

        pinto = AlgorithmPinto()
        # print(pinto.height_node(nx.bfs_tree(g, source=1), 1, 2))
        # pinto.Algorithm(g, [2, 3], 0, 1)
        # print("Source of rumor is %s" % source_estimation)