# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:10:28 2015

@author: h

Computes a random infection through a Graph.
the parameters are:

G : the graph
seeds : an array containing the seeds
ratio : the proportion of the graph at which the algorithm should stop when it has been infected
prob : the probability of transmitting the infection
"""
import networkx as nx
import numpy as np


class Infection:

    def __init__(self):
        pass

    def run(G, seeds, ratio, prob):
        frontier = nx.Graph()
        infected = nx.Graph()

        for seed in seeds:
            infected.add_node(seed)
        for node in infected:
            for neighbor in G.neighbors(node):
                if not infected.has_node(neighbor):
                    frontier.add_node(neighbor)
        seeds = []
        while len(infected.node) < ratio * len(G.node):
            Infection.ripple_step(G, frontier, infected, prob)

        Infection.edging(G, infected)

        return infected

    def ripple_step(G, frontier, infected, prob):
        """ Generating a step in the infection simulation"""

        # List containing the nodes from the frontier at time t with k
        # infected neighbors
        frontier_degree_t = []

        # List containing the newly infected nodes
        infected_t = []

        # Generating the lists containing the nodes with i infected neighbors
        for node in frontier:
            i = 0
            for neighbor in G.neighbors(node):
                if infected.has_node(neighbor):
                    i += 1
            try:
                frontier_degree_t[i-1].append(node)
            except IndexError:
                for j in range(0, i+1):
                    t = []
                    frontier_degree_t.append(t)
                frontier_degree_t[i-1].append(node)

        # Generating the optimal step for the infection
        for j in range(0, len(frontier_degree_t)):
            if len(frontier_degree_t[j]) > 0:
                f_j = len(frontier_degree_t[j])
                p_j = 1-(1-prob)**(j+1)
                n_j = int(min(np.floor(p_j*(f_j+1)), f_j))
                # The f_j/prob implicates n_j > f_j

                infected_t.append(np.random.choice(frontier_degree_t[j], n_j,
                                  replace=False))

        # Updating the frontier
        for j in infected_t:
            for node in j:
                infected.add_node(node)
                frontier.remove_node(node)
                for neighbor in G.neighbors(node):
                    if not infected.has_node(neighbor):
                        frontier.add_node(neighbor)
        return

    def edging(graph, graph_i):
        """ Transposes the edges from graph to graph_i"""
        for node in graph_i:
            for neighbor in graph.neighbors(node):
                if graph_i.has_node(neighbor):
                    graph_i.add_edge(node, neighbor)
