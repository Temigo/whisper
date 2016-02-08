# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 18:54:57 2016

@author: h
"""

# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# Implementation of the Netsleuth algorithm

# from graph import Node, Edge, Graph
import networkx as nx
import numpy as np
from scipy.misc import comb
from sympy import N as numeric
from numpy import zeros
from scipy.sparse.linalg import eigs as eigs
from scipy import linalg
from sympy import Matrix


class AlgorithmNetsleuth:
    def __init__(self):
        pass

    # graph is the graph, constitued from i_graph_init containing the initially
    #       infected nodes and frontier containing the neighbours of i_graph
    # prob is the infection probability
    #
    # seeds contains the susceptible sources
    #
    # i_graphe containing the considered infected nodes

    #####################################################################
    # ATTRIBUTES OF THE NODE :
    #
    # name i : node number i (0 <= i < n)
    #       has to be in common between graph and graphi
    #
    # informed : 0 = non informed  (SEE if needed)
    #            1 = informed
    #
    # Example : graph.add_node(1, informed = 0)
    #
    #####################################################################

    def run(self, graph, i_graph_init, prob):
        """Executes the Netsleuth algotithm on graph, given the infected nodes
        i_graph_init from graph"""
        # Initiating the seeds
        seeds = []
        number_of_seeds = 0
        frontier = nx.Graph()
        for node in i_graph_init:
            clear_neighbors = len(graph.neighbors(node)) -\
                                  len(i_graph_init.neighbors(node))
            if clear_neighbors > 0:
                frontier.add_node(node, clear=clear_neighbors)

        # The infected nodes for computing the first seed are the true ones
        i_graph = nx.Graph(i_graph_init)

        # The Minimal Description Length criteria
        decreasing_description_length = True
        previous_encoding = 0

        while decreasing_description_length:

            # Getting the next better seed
            seed = self.etape(frontier, i_graph)
            seeds.append(seed)

            # Calculating the MDL for the seed set
            seed_encoding = (self.logn(len(seeds)) +
                             np.log2(comb(len(i_graph_init.nodes()),
                                          len(seeds))))
            # Getting the most truthworthly ripple
            ripple_encoding = self.ripple(i_graph_init, seeds, prob)

            # Removing the seed for the calculation of the next one
            i_graph.remove_node(seed)

            # Calculating the actual Minimal Description Length (MDL)
            total_encoding = seed_encoding + ripple_encoding
            if previous_encoding == 0 or previous_encoding > total_encoding:
                previous_encoding = total_encoding
            else:
                number_of_seeds = len(seeds) - 1
                decreasing_description_length = False

            frontier.clear()
            for node in i_graph:
                clear_neighbors = len(graph.neighbors(node)) -\
                                      len(i_graph.neighbors(node))
                if clear_neighbors > 0:
                    frontier.add_node(node, clear=clear_neighbors)

        return seeds[:number_of_seeds]

    def etape(self, frontier, i_graph):
        """ Calculates the most probable seed within the infected nodes"""

        # Taking the actual submatrix, not the laplacian matrix. The change
        # lies in the total number of connections (The diagonal terms) for the
        # infected nodes connected to uninfected ones in the initial graph
        i_laplacian_matrix = nx.laplacian_matrix(i_graph)
        for i in range(0, len(i_graph.nodes())):
            if frontier.has_node(i_graph.nodes()[i]):
                i_laplacian_matrix[i, i] +=\
                                frontier.node[i_graph.nodes()[i]]['clear']

        # SymPy
        Lm = Matrix(i_laplacian_matrix.todense())
        i = self.Sym2NumArray(Matrix(Lm.eigenvects()[0][2][0])).argmax()

        # NumPy
        # val, vect = linalg.eigh(i_laplacian_matrix.todense())
        #i = vect[0].argmax()

        # SciPY
        # val, vect = eigs(i_laplacian_matrix.rint())
        # i = vect[:, 0].argmax()

        seed = (i_graph.nodes()[i])

        return seed

    def ripple(self, i_graph_init, seeds, prob):
        """ Simulating the most truthworthly ripple from the seed set"""

        # The uninfected nodes adjacent to the infected ones
        frontier = nx.Graph()

        # The already infected nodes in the ripple
        infected = nx.Graph()
        # Initiating the infected graph
        for seed in seeds:
            infected.add_node(seed)
        for seed in seeds:
            for neighbor in i_graph_init.neighbors(seed):
                if not infected.has_node(neighbor):
                    frontier.add_node(neighbor)

        ripple_encoding = 0
        step = 0

        # Generating the ripple
        while len(infected.nodes()) != len(i_graph_init.nodes()):
            if len(frontier.nodes()) == 0:
                break
            step += 1

            # Calculating the infected nodes at next time step
            step_encoding = self.ripple_step(i_graph_init,
                                                  frontier,
                                                  infected,
                                                  prob)
            ripple_encoding += step_encoding

        return ripple_encoding + self.logn(step)

    @staticmethod
    def ripple_step(i_graph_init, frontier, infected, prob):
        """ Generating a step in the infection simulation"""

        # Linst containing the nodes from the frontier at time t with k
        # infected neighbors
        frontier_degree_t = []

        # List containing the newly infected nodes
        infected_t = []

        step_encoding = 0

        # Generating the lists containing the nodes with i infected neighbors
        for node in frontier:
            i = 0
            for neighbor in i_graph_init.neighbors(node):
                if infected.has_node(neighbor):
                    i += 1
            try:
                frontier_degree_t[i-1].append(node)
            except IndexError:
                for j in range(0, i+1):
                    t = []
                    frontier_degree_t.append(t)
                frontier_degree_t[i-1].append(node)

        # Generating the optimal step for the ripple
        for j in range(0, len(frontier_degree_t)):
            if len(frontier_degree_t[j]) > 0:
                f_j = len(frontier_degree_t[j])
                p_j = 1-(1-prob)**(j+1)
                n_j = int(min(np.floor(p_j*(f_j+1)), f_j))
                if prob < 1:
                    # The f_j/prob implicates n_j > f_j
                    infected_t.append(np.random.choice(frontier_degree_t[j],
                                                       n_j, replace=False))
                else:
                    infected_t.append(frontier_degree_t[j])

                step_encoding -= (np.log2(comb(f_j, n_j) * (p_j ** n_j) *
                                          (1-p_j) ** (f_j - n_j)) +
                                  n_j*np.log2(n_j/f_j))
                if f_j != n_j:
                    step_encoding += (f_j - n_j)*np.log2(1-n_j/f_j)

        # Updating the frontier
        for j in infected_t:
            for node in j:
                infected.add_node(node)
                frontier.remove_node(node)
                for neighbor in i_graph_init.neighbors(node):
                    if not infected.has_node(neighbor):
                        frontier.add_node(neighbor)

        return step_encoding

    @staticmethod
    def logn(a):
        if a == 0:
            return 0
        s = 0
        b = a
        while np.log2(b) > 0:
            b = np.log2(b)
            s += b
        return s + np.log2(2.865064)

    @staticmethod
    def Sym2NumArray(F):
        shapeF = F.shape
        B = zeros(shapeF)
        for i in range(0, shapeF[0]):
            for j in range(0, shapeF[1]):
                B[i, j] = numeric(F[i, j])
        return B
