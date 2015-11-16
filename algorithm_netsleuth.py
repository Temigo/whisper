# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - Netsleuth algorithm
    
# My actual code :
# The last unindented part was for testing : I took a disc on a grid for the infected graph
    
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


class AlgorithmNetsleuth:
    def __init__(self):
        pass

    # graph is the general graph. It is currently not used.
    # i_graph_init is the graph of the infected nodes.
    # prob is the probability for an infected node to transmit it to a neighbor.

    def run(self, graph, i_graph_init, prob):
        """Executes the Netsleuth algotithm on graph, given the infected nodes
        i_graph_init from graph"""
        # Initiating the seeds
        seeds = []

        # Needed to manage the uninfected nods in the laplacian matrix
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
            seed = self.etape(i_graph)
            seeds.append(seed)

            # Calculating the MDL for the seed set
            seed_encoding = (self.logn(len(seeds)) +
                             np.log2(comb(len(i_graph_init.nodes()), len(seeds))))
            # Getting the most truthworthly ripple
            ripple_encoding = self.ripple(i_graph_init, seeds, prob)

            # Removing the seed for the calculation of the next one
            i_graph.remove_node(seed)

            # Calculating the actual Minimal Description Length (MDL)
            total_encoding = seed_encoding + ripple_encoding
            if previous_encoding == 0 or previous_encoding > total_encoding:
                previous_encoding = total_encoding
            else:
                number_of_seeds = len(seeds) - 1  # FIXME : to be defined before else statement
                decreasing_description_length = False

            frontier.clear()
            for node in i_graph:
                clear_neighbors = len(graph.neighbors(node)) -\
                                      len(i_graph.neighbors(node))
                if clear_neighbors > 0:
                    frontier.add_node(node, clear=clear_neighbors)

        return seeds[:number_of_seeds]

    @staticmethod
    def etape(i_graph):
        """ Calculates the most probable seed within the infected nodes"""

        # Taking the actual submatrix, not the laplacian matrix. The change lies in the total number of connections
        # (The diagonal terms) for the infected nodes connected to uninfected ones in the initial graph
        # TODO frontier is undefined
        i_laplacian_matrix = nx.laplacian_matrix(i_graph)
        for i in range(0, len(i_graph.nodes())):
            if frontier.has_node(i_graph.nodes()[i]):
                i_laplacian_matrix[i, i] +=\
                                frontier.node[i_graph.nodes()[i]]['clear']

        # Getting the node with the highest coordinate for the eigenvector of the smallest eigenvalue:
        
        # Lm = Matrix(i_laplacian_matrix.todense())
        # i = Netsleuth.Sym2NumArray(Matrix(Lm.eigenvects()[0][2][0])).argmax()

        # val, vect = linalg.eigh(i_laplacian_matrix.todense())
        # i = vect[0].argmax()

        # Working good and fast !
        val, vect = eigs(i_laplacian_matrix.rint())
        i = vect[:, 0].argmax()

        seed = (i_graph.nodes()[i])

        return seed

    def ripple(self, i_graph_init, seeds, prob):
        """ Simulating the most truthworthly ripple from the seed set and returns the MDL"""

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
        """ Generating a step in the ripple simulation"""

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
                n_j = int(min(np.floor(p_j*(f_j+1)), f_j))  # The f_j/prob implicates n_j > f_j
                infected_t.append(np.random.choice(frontier_degree_t[j], n_j,
                                  replace=False))

                step_encoding -= (np.log2(comb(f_j, n_j) * (p_j ** n_j) *
                                          (1-p_j) ** (f_j - n_j)) +
                                  n_j*np.log2(n_j/f_j))
                if f_j != n_j:
                    step_encoding += (f_j - n_j)*np.log2(1-n_j/f_j)

                try:
                    for node in infected_t[j]:
                        infected.add_node(node)
                except IndexError:  # Due to empty d_class frontier sets
                    pass

        # Updating the frontier
        for j in infected_t:
            for node in j:
                frontier.remove_node(node)
                for neighbor in i_graph_init.neighbors(node):
                    if not infected.has_node(neighbor):
                        frontier.add_node(neighbor)

        return step_encoding

    @staticmethod
    def logn(a):
        """ Calculating the MDL for a integer"""
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
        """ For convertion from sympy to numpy matrix"""
        shapeF = F.shape
        B = zeros(shapeF)
        for i in range(0, shapeF[0]):
            for j in range(0, shapeF[1]):
                B[i, j] = numeric(F[i, j])
        return B
