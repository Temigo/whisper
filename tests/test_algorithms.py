# -*- coding: utf-8 -*-
# Tests of algorithms

from algorithm_shah_zaman import AlgorithmSZ
from algorithm_pinto import AlgorithmPinto
from algorithm_netsleuth import AlgorithmNetsleuth
import unittest
import networkx as nx


class TestSZ(unittest.TestCase):

    def test_elementary(self):
        """
            1
          /  \
         2   3
        :return:
        """
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3])
        g.add_edges_from([(1, 2), (1, 3)])

        g[1]['infected'] = True
        g[2]['infected'] = True
        g[3]['infected'] = True

        sz = AlgorithmSZ()
        source_estimation = sz.run(g)
        # self.assertEqual()
        print(source_estimation)
        self.assertEqual(source_estimation, 1)


class TestPinto(unittest.TestCase):
    pass


class TestNetsleuth(unittest.TestCase):
    pass