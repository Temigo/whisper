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
			 /      \
			2        3
		   / \     
		   4  5
		:return:
		"""
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edges_from([(1, 2), (2, 1), (1, 3), (3, 1), (2, 4), (4, 2), (5, 2), (2, 5)])
        g.node[1]['infected'] = True
        g.node[2]['infected'] = True
        g.node[4]['infected'] = True
        g.node[5]['infected'] = True

        g.node[4]['time'] = 2.
        g.node[5]['time'] = 2.
        g.node[2]['time'] = 1.

        pinto = AlgorithmPinto()
        # print(pinto.height_node(g, 5, 3))
        source = pinto.run(g, [5, 4, 2], 1, 1)
        print("Source of rumor is %s" % source)

    def test_graph(self):
        """
        1          5
        | \      /  \
        |  3 -- 4   6 - 8 - 9 - 10 - 11
        | /     \  /
        2        7
        :return:
        """
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        g.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4), (4,5), (5, 6), (6, 7), (7, 4), (6, 8), (8, 9), (9, 10), (10, 11)])

        g.node[1]['infected'] = True
        g.node[2]['infected'] = False
        g.node[3]['infected'] = True
        g.node[4]['infected'] = True
        g.node[5]['infected'] = True
        g.node[6]['infected'] = True
        g.node[7]['infected'] = False
        g.node[8]['infected'] = False
        g.node[9]['infected'] = True

        g.node[1]['time'] = 2
        g.node[3]['time'] = 1
        g.node[5]['time'] = 1
        g.node[6]['time'] = 2

        pinto = AlgorithmPinto()
        source_estimation = pinto.run(g, [1, 3, 5, 6], 1, 1)
        # self.assertEqual(source_estimation, 3)
        print("Source of rumor is %s" % source_estimation)
