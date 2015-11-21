# -*- coding: utf-8 -*-
# Tests of algorithm FC

from algorithm_fioriti_chinnici import AlgorithmFC
import networkx as nx
import unittest


class TestFC(unittest.TestCase):
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

        fc = AlgorithmFC()
        print(fc.run(g))
        # pinto.Algorithm(g, [2, 3], 0, 1)
        # print("Source of rumor is %s" % source_estimation)

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

        g.node[1]['infected'] = False
        g.node[3]['infected'] = False
        g.node[4]['infected'] = True
        g.node[5]['infected'] = True
        g.node[2]['infected'] = True

        fc = AlgorithmFC()
        print(fc.run(g))