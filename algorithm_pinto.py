# Algorithms used for rumor source inference in a graph
# - Pinto algorithm

import networkx as nx
import numpy
from numpy import matrix, array
from numpy import linalg
import math

#####################################################################
# ATRIBUTES OF THE NODE:
#
# name: i = node number i (0 < i < n)
#
# informed: 0 = ignorant
#           1 = informed/infected
#
# observer: 0 = non observer
#           i = observer number i (0 < i < k)
#
# path: i = number of edges between the node and the
# supposed source (SEE IF USEFULL)
#
# time: t = time when the observer receives the information
#
# bfs: auxiliar boolean to construct the bfs tree
#
# children: children nodes for the bfs tree
#
# Example: G.add_node(1,informed=0,observer=0,path=0,time=0)
#####################################################################


# O is a vector with the observer nodes
# propagation delays are RVs with Gaussian distribution N(mi,sigma2)
class AlgorithmPinto:
    def __init__(self):
        pass

    def run(self, G, O, mi, sigma2):
        """
        Main
        :param G: graph
        :param O: list of observers <<<< ACTIVE observers !
        :param mi: mean
        :param sigma2: variance
        :return:
        """
        # TODO : consider only active observers !
        first_node = O[0]

        # Compute the delay vector d relative to first_node
        d = self.observed_delay(G, O)

        # calculates F for the first node: fulfills max
        max = self.main_function(first_node, O, d, nx.bfs_tree(G, source=first_node), mi, sigma2)

        source = first_node  # SEE HOW WE CAN DO IT
        # calculates the maximum F
        for s in G:  # FIXME is this G_a ?
            # Compute the spanning tree rooted at s
            T = nx.bfs_tree(G, source=s)
            F = self.main_function(s, O, d, T, mi, sigma2)

            if F > max:
                max = F
                source = s
        return source

    # MAIN_FUNCTION S to be calculated
    def main_function(self, s, O, d, T, mi, sigma2):
        """

        :param s: source (hypothesis)
        :param O: list of observers
        :param d: delay vector
        :param T: tree (bfs)
        :param mi: mean
        :param sigma2: variance
        :return:
        """
        mu_s = self.deterministic_delay(T, s, O, mi)
        delta = self.delay_covariance(T, O, sigma2)
        inverse = numpy.linalg.inv(delta)
        return (mu_s.T * inverse * (d - (0.5 * mu_s)))[0, 0]

    # calculates array d (observed delay)
    @staticmethod
    def observed_delay(g, O):
        d = numpy.zeros(shape=(len(O)-1, 1))
        for i in range(len(O) - 1):
            d[i][0] = g.node[O[i + 1]]['time'] - g.node[O[i]]['time']
        return d

    # calculates array mi_s (deterministic delay)
    def deterministic_delay(self, T, s, O, mi):
        """
        Computes mu_s
        :param T: tree
        :param s: source
        :param O: list of observers
        :param mi: mean
        :return:
        """
        constant = self.height_node(T, s, O[0])
        mi_s = numpy.zeros(shape=(len(O)-1, 1))
        for i in range(len(O)-1):
            mi_s[i][0] = self.height_node(T, s, O[i + 1]) - constant
        mi_s = mi * mi_s
        return mi_s

    # calculates the height of a node in the tree T (recursive)
    def height_node(self, T, s, node):
        l = list(nx.all_simple_paths(T, s, node))
        if l == []:
            return 0
        else:
            return len(l[0]) - 1

    # calculates the array delta (delay covariance)
    def delay_covariance(self, T, O, sigma2):
        """
        Computes lambda
        :param T: tree
        :param O: list of observers
        :param sigma2: variance
        :return:
        """
        # TODO stop using all_simple_paths (complexity)
        n = len(O)
        delta = numpy.zeros(shape=(n-1, n-1))
        T = T.to_undirected()
        for k in range(n-1):
            for i in range(n-1):
                if i == k:
                    delta[k][i] = len(list(nx.all_simple_paths(T, O[0], O[k+1]))[0]) - 1
                else:
                    c1 = list(nx.all_simple_paths(T, O[0], O[k+1]))[0]
                    c2 = list(nx.all_simple_paths(T, O[0], O[i+1]))[0]
                    S = [x for x in c1 if x in c2]
                    delta[k][i] = len(S) - 1
        delta = delta * (sigma2 ** 2)  # FIXME : square or not ?
        return delta
