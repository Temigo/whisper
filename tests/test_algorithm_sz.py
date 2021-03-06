# -*- coding: utf-8 -*-
# Tests of algorithm Shah and Zaman

from algorithm_shah_zaman import AlgorithmSZ
import networkx as nx
import unittest
import matplotlib.pyplot as plt


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
        source_estimation = sz.run(g, g)
        self.assertEqual(source_estimation, 1)

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

        g_i = nx.Graph()
        g_i.add_nodes_from([2, 4, 5])
        g_i.add_edges_from([(2, 4), (2, 5)])

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, g_i)
        self.assertEqual(source_estimation, 2)
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

        g_i = nx.Graph()
        g_i.add_nodes_from([1, 2, 3, 4])
        g_i.add_edges_from([(1, 2), (2, 3), (3, 1), (3, 4)])

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, g_i)
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
        source_estimation = sz.run(g, g)
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
        source_estimation = sz.run(g, g)
        print("Source of rumor is %s" % source_estimation)

        # nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        # plt.show()

    def test_other_graph(self):
        g = nx.chvatal_graph()
        nx.set_node_attributes(g, 'infected', {n: False for n in g.nodes()})
        g.node[0]['infected'] = True

        g_i = nx.Graph()
        g_i.add_nodes_from([0])

        sz = AlgorithmSZ()
        source_estimation = sz.run(g, g_i)
        print("Source of rumor is %s" % source_estimation)

        # nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        # plt.show()
