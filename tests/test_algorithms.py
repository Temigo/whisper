# -*- coding: utf-8 -*-
# Tests of algorithms

from algorithm_shah_zaman import AlgorithmSZ
from algorithm_pinto import AlgorithmPinto
from algorithm_netsleuth import AlgorithmNetsleuth
from algorithm_fioriti_chinnici import AlgorithmFC

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

        # nx.draw_networkx(g, node_color=['b' if g.node[n]['infected'] else 'r' for n in g])
        # plt.show()


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


class TestNetsleuth(unittest.TestCase):
    @staticmethod
    def plot(G):
        """ A nice plotting function for the graphs"""

        pos = nx.spring_layout(G, iterations=1000)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_nodes(G, pos, node_size=500)

    @staticmethod
    def i_plot(G, G_i):
        """ A nice plotting function for the graphs"""

        pos = nx.spring_layout(G, iterations=500)
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='g')
        nx.draw_networkx_nodes(G_i, pos, node_size=500, node_color='r')

    @staticmethod
    def edging(graph, graph_i):
        for node in graph_i:
            for neighbor in graph.neighbors(node):
                if graph_i.has_node(neighbor):
                    graph_i.add_edge(node, neighbor)

    def test_disc(self):
        G = nx.Graph()

        l = 27

        # for i in range(0, l**2):
        #    G.add_node(i, name=i)

        for i in range(0, l-1):
            for j in range(0, l-1):
                if abs((l-1)/2-i)+abs((l-1)/2-j) <= (l-1)/2 and abs((l-1)/2-(i+1)) \
                        + abs((l-1)/2-j) <= (l-1)/2:
                    a = i+l*j
                    b = +1+l*j
                    G.add_node(100+i+l*j, name=100+i+l*j)
                    G.add_node(100+i+1+l*j, name=100+i+1+l*j)
                    G.add_edge(100+i+l*j, 100+i+1+l*j)
                else:
                    pass  # G.add_edge(i+l*j, i+l*j)

        for i in range(0, l-1):
            for j in range(0, l-1):
                if abs((l-1)/2-i)+abs((l-1)/2-j) <= (l-1)/2 and abs((l-1)/2-i-1) \
                        + abs((l-1)/2-j) <= (l-1)/2:
                    a = i*l+j
                    b = i*l+l+j
                    G.add_node(100+i*l+j, name=100+i*l+j)
                    G.add_node(100+(i+1)*l+j, name=100+(i+1)*l+j)
                    G.add_edge(100+i*l+j, 100+(i+1)*l+j)
                else:
                    pass  # G.add_edge(i*l+j, (i)*l+j)
        """
        for i in range(0, l-1):
            for j in range(0, l-1):
                if abs((l-1)/2-i)+abs((l-1)/2-j) <= (l-1)/2 and abs((l-1)/2-(i+1)) \
                        + abs((l-1)/2-j) <= (l-1)/2:
                    a = i+l*j
                    b = +1+l*j
                    G.add_node(1000+i+l*j, name=1000+i+l*j)
                    G.add_node(1000+i+1+l*j, name=1000+i+1+l*j)
                    G.add_edge(1000+i+l*j, 1000+i+1+l*j)
                else:
                    pass  # G.add_edge(i+l*j, i+l*j)

        for i in range(0, l-1):
            for j in range(0, l-1):
                if abs((l-1)/2-i)+abs((l-1)/2-j) <= (l-1)/2 and abs((l-1)/2-i-1) \
                        + abs((l-1)/2-j) <= (l-1)/2:
                    a = i*l+j
                    b = i*l+l+j
                    G.add_node(1000+i*l+j, name=1000+i*l+j)
                    G.add_node(100+(i+1)*l+j, name=1000+(i+1)*l+j)
                    G.add_edge(1000+i*l+j, 1000+(i+1)*l+j)
                else:
                    pass  # G.add_edge(i*l+j, (i)*l+j)

        G.add_edge(156, 1056)"""

        G_i = nx.Graph()
        for node in G.nodes():
            if len(G.neighbors(node)) >= 4:
                G_i.add_node(node)
        self.edging(G, G_i)

        self.i_plot(G, G_i)

        graph = nx.Graph(G)
        i_graph_init = nx.Graph(G_i)
        netsleuth = AlgorithmNetsleuth()
        s = netsleuth.run(G, G_i, 0.5)
        print(s)

        # plot(G)


        # infected = nx.Graph()
        # infected.add_node(320)
        # frontier = nx.Graph()
        # for neighbor in G.neighbors(320):
        #    frontier.add_node(neighbor)
        #
        # b = 0.001
        # j = 1
        # n = 1
        # p = 1-(1-b)**j
        # np.floor(p*(n/b+1))/n


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