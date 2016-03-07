# -*- coding: utf-8 -*-
# Tests of algorithm Netsleuth

from algorithm_netsleuth import AlgorithmNetsleuth
import networkx as nx
import unittest
import matplotlib.pyplot as plt


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
        plt.show()

    @staticmethod
    def edging(graph, graph_i):
        for node in graph_i:
            for neighbor in graph.neighbors(node):
                if graph_i.has_node(neighbor):
                    graph_i.add_edge(node, neighbor)

    def test_disc(self):
        G = nx.Graph()

        l = 7

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

        #

        graph = nx.Graph(G)
        i_graph_init = nx.Graph(G_i)
        netsleuth = AlgorithmNetsleuth()
        s = netsleuth.run(G, G_i, 0.5)
        print(s)
        # self.i_plot(G, G_i)

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
