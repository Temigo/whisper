# -*- coding: utf-8 -*-
# Tests of algorithm Pinto

from pinto_algorithm import AlgorithmPinto
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
		g.add_edges_from([(1,2),(2,1),(1,3),(3,1),(2,4),(4,2),(5,2),(2,5)])
		g.node[1]['infected'] = True
		g.node[2]['infected'] = True
		g.node[4]['infected'] = True
		g.node[5]['infected'] = True
		
		g.node[4]['time'] = 2.
		g.node[5]['time'] = 2.
		g.node[2]['time'] = 1.

		pinto = AlgorithmPinto()
		#print(pinto.height_node(g, 5, 3))
		pinto.Algorithm(g, [2,4,5], 0, 1)
		#print("Source of rumor is %s" % source_estimation)
