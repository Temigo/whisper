# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - Netsleuth algorithm

from graph import Node, Edge, Graph


class AlgorithmNetsleuth:
    def __init__(self):
        pass
    
    
    
# My actual code :
# The last unindented part was for testing : I took a disc on a grid for the infected graph
    
# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# Implementation of the Netsleuth algorithm

# from graph import Node, Edge, Graph
import networkx as nx
from scipy import linalg
import numpy as np
from scipy.misc import comb


class Netsleuth:
    def __init__(self):
        pass

    # graph is the general graph. It is currently not used.
    # i_graph_init is the graph of the infected nodes.
    # prob is the probability for an infected node to transmit it to a neighbor.

    def run(graph, i_graph_init, prob):
        """Executes the Netsleuth algotithm on i_graph_init, the graph of the infected nodes"""
        # Initiating the seeds
        seeds = []

        # The infected nodes for computing the first seed are the true ones
        i_graph = nx.Graph(i_graph_init)

        # The Minimal Description Length criteria
        decreasing_description_length = True
        previous_encoding = 0

        while decreasing_description_length:

            # Getting the next better seed
            seed = Netsleuth.etape(i_graph)
            seeds.append(seed)

            # Calculating the MDL for the seed set
            seed_encoding = (Netsleuth.logn(len(seeds)) +
                             np.log2(comb(len(i_graph_init.nodes()), len(seeds))))
            # Getting the most truthworthly ripple
            ripple_encoding = Netsleuth.ripple(i_graph_init, seeds, prob)

            # Removing the seed for the calculation of the next one
            i_graph.remove_node(seed)

            # Calculating the actual Minimal Description Length (MDL)
            total_encoding = seed_encoding + ripple_encoding
            if previous_encoding == 0 or previous_encoding > total_encoding:
                previous_encoding = total_encoding
            else:
                number_of_seeds = len(seeds) - 1
                decreasing_description_length = False

        return seeds[:number_of_seeds]

    def etape(i_graph):
        """ Calculates the most probable seed within the infected nodes"""

        i_laplacian_matrix = nx.laplacian_matrix(i_graph)
        # Diagonalizing the laplacian matrix
        val, vect = linalg.eigh(i_laplacian_matrix.todense()) # The eigh function gives wrong eigenvects for eigenval 0

        # Getting the node with the highest coordinate for the eigenvector of the smallest eigenvalue
        i = abs(vect[0]).argmax()

        seed = (i_graph.nodes()[i])

        return seed

    def ripple(i_graph_init, seeds, prob):
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
            if len(frontier.nodes())==0:
                break;
            step += 1

            # Calculating the infected nodes at next time step
            step_encoding = Netsleuth.ripple_step(i_graph_init,
                                                  frontier,
                                                  infected,
                                                  prob)
            ripple_encoding += step_encoding

        return ripple_encoding + Netsleuth.logn(step)

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
                n_j = int(min(np.floor(p_j*(f_j/prob+1)), f_j)) # The f_j/prob implicates n_j > f_j
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
                except IndexError: # Due to empty d_class frontier sets
                    pass

        # Updating the frontier
        for j in infected_t:
            for node in j:
                frontier.remove_node(node)
                for neighbor in i_graph_init.neighbors(node):
                    if not infected.has_node(neighbor):
                        frontier.add_node(neighbor)

        return step_encoding

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

# for testing


def plot(G):
    """ A nice plotting function for the graphs"""

    pos = nx.spring_layout(G, iterations=1000)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_nodes(G, pos, node_size=500)


def edging(graph, graph_i):
    """ Transposes the edges from graph to graph_i"""
    for node in graph_i:
        for neighbor in graph.neighbors(node):
            if graph_i.has_node(neighbor):
                graph_i.add_edge(node, neighbor)

# Creating a disc on a grid : for l=5, with I for infected and O for clear
#
#  O--O--I--O--O
#  |  |  |  |  |
#  O--I--I--I--O
#  |  |  |  |  |
#  I--I--I--I--I
#  |  |  |  |  |
#  O--I--I--I--O
#  |  |  |  |  |
#  O--O--I--O--O  
#
G = nx.Graph()

l = 6

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

plot(G)
s = Netsleuth.run(G, G, 0.5)
print(s)
