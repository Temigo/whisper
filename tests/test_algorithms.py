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
        g.add_edges_from([(1, 2), (1, 3)])  # FIXME do you need to add also (2, 1) and (3, 1) ?

        g[1]['infected'] = True
        g[2]['infected'] = True
        g[3]['infected'] = True

        sz = AlgorithmSZ()
        # Test various v options (root of the spanning tree)
        source_estimation = sz.run(g, v=1)
        self.assertEqual(source_estimation, 1)
        source_estimation2 = sz.run(g, v=2)
        self.assertEqual(source_estimation2, 1)
        source_estimation3 = sz.run(g, v=3)
        self.assertEqual(source_estimation3, 1)

        print("Source of rumor is %s" % source_estimation)

    def test_less_elementary(self):
        """
              1
             / \
            2  3
           / \
          4  5
        :return:
        """
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])

        g[1]['infected'] = False
        g[3]['infected'] = False
        g[4]['infected'] = True
        g[5]['infected'] = True
        g[2]['infected'] = True

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=2)
        self.assertEqual(source_estimation, 2)
        source_estimation2 = sz.run(g, v=4)
        self.assertEqual(source_estimation2, 2)
        source_estimation3 = sz.run(g, v=5)
        self.assertEqual(source_estimation3, 2)
        print("Source of rumor is %s" % source_estimation)


class TestPinto(unittest.TestCase):
    pass


class TestNetsleuth(unittest.TestCase):
    pass