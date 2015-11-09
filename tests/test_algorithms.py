# -*- coding: utf-8 -*-
# Tests of algorithms

from algorithm_shah_zaman import AlgorithmSZ
from algorithm_pinto import AlgorithmPinto
from algorithm_netsleuth import AlgorithmNetsleuth

import matplotlib.pyplot as plt
import networkx as nx
import unittest


class TestSZ(unittest.TestCase):

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

        g.node[1]['infected'] = False
        g.node[3]['infected'] = False
        g.node[4]['infected'] = True
        g.node[5]['infected'] = True
        g.node[2]['infected'] = True

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=2)
        self.assertEqual(source_estimation, 2)
        source_estimation2 = sz.run(g, v=4)
        self.assertEqual(source_estimation2, 2)
        source_estimation3 = sz.run(g, v=5)
        self.assertEqual(source_estimation3, 2)
        print("Source of rumor is %s" % source_estimation)

    def test_graph(self):
        """
        1          5
        | \      /  \
        |  3 -- 4   6
        | /     \  /
        2        7
        :return:
        """
        g = nx.Graph()
        g.add_nodes_from([1, 2, 3, 4, 5, 6, 7])
        g.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4), (4,5), (5, 6), (6, 7), (7, 4)])

        g.node[1]['infected'] = True
        g.node[2]['infected'] = True
        g.node[3]['infected'] = True
        g.node[4]['infected'] = True
        g.node[5]['infected'] = False
        g.node[6]['infected'] = False
        g.node[7]['infected'] = False

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=4)
        self.assertEqual(source_estimation, 3)
        print("Source of rumor is %s" % source_estimation)

    def test_random_graph(self):
        g = nx.fast_gnp_random_graph(10, 0.5)
        nx.set_node_attributes(g, 'infected', {n: False for n in g.nodes()})
        g.node[1]['infected'] = True
        g.node[2]['infected'] = True
        g.node[3]['infected'] = True
        g.node[4]['infected'] = True

        # FIXME Cannot assert anything because graph is random !
        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=4)
        print("Source of rumor is %s" % source_estimation)

        # Graph drawing
        # nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        # plt.show()

    def test_wheel_graph(self):
        g = nx.wheel_graph(10)
        nx.set_node_attributes(g, 'infected', {n: False for n in g.nodes()})
        g.node[0]['infected'] = True
        g.node[1]['infected'] = True
        g.node[5]['infected'] = True

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=1)
        print("Source of rumor is %s" % source_estimation)

        # nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        # plt.show()

    def test_other_graph(self):
        g = nx.chvatal_graph()
        nx.set_node_attributes(g, 'infected', {n: False for n in g.nodes()})
        g.node[0]['infected'] = True

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, v=0)
        print("Source of rumor is %s" % source_estimation)

        nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        plt.show()


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
        print(pinto.height_node(nx.bfs_tree(g, source=1), 1, 2))
        # pinto.Algorithm(g, [2, 3], 0, 1)
        # print("Source of rumor is %s" % source_estimation)


class TestNetsleuth(unittest.TestCase):
    pass