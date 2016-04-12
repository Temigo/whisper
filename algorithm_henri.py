# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:27:14 2015

@author: h
"""

import networkx as nx
import numpy as np


class Custom:
    def __init__(self):
        pass

    def run(G, G_i, prob, j=10, Offset=3):
        """
        The actual algorithm is executed in the run_one function.
        This function implements a filter on the outputs of the real algorithm
        It executes it several times, gathering the outputs and then works out
        a best center seed for each group of seeds close enough to each other.

        The actual algorithm is simply a kind of BFS on the infected subgraph,
        starting from the border and getting through the nodes one layer of
        nodes after the other following an pattern as the ripple of an
        infection, selecting as source the nodes were the layers collapse on
        one-self.

        The sources are then filtered a first time using the same method as in
        run to collapse the remanants of each layer into one seed

        The parameters are :
            G the general graph,
            G_i the infected subgraph,
            prob the probability of each infected node to infect one of its
        neighbor at each time-step
            j the number of iterations the filtering function should do in the
        harmonization process of run(). The execution-time is directly
        multiplicated by j
            Offset is the term used to harmonize the seeds : the greater, the
        higher two distant seeds will be associated together
        """
        ss = []  # To strore the seeds
        for i in range(0, j):  # Gathering the seeds from the actual algorithm
            for seed in Custom.run_once(G, G_i, prob):
                ss.append(seed)
        if j > 1:  # The harmonisation
            i = 0
            # Creating an union-find structure for the connex seed-parts
            nb = []  # Used to store the number of seeds in each connex part
            classes = []  # The union-find structure
            sources = []
            indexToScan = []  # Wich connex part is next to be scanned

            # The Graph structure is a convienent Hash-table
            G_seeds = nx.Graph()
            todo = nx.Graph()

            # Innitiating the union-find structure
            for seed in ss:
                G_seeds.add_node(seed, {'class': -1})
            for seed in G_seeds.nodes():  # Getting the edges between seeds
                for neighbor in G_i.neighbors(seed):
                    if G_seeds.has_node(neighbor):
                        G_seeds.add_edge(seed, neighbor)

            # The connex parts are computed through a BFS in G_seed
            for seed in G_seeds.nodes():
                if G_seeds.node[seed]['class'] != -1:  # seed already was seen
                    continue
                else:
                    G_seeds.node[seed]['class'] = i
                    classes.append(i)
                    indexToScan.append(i)
                    nb.append(1)
                    for neighbor in G_seeds.neighbors(seed):
                        todo.add_node(neighbor)
                    while(len(todo.nodes()) > 0):  # The BFS
                        for node in todo.nodes():
                            todo.remove_node(node)
                            if G_seeds.node[node]['class'] == -1:
                                G_seeds.node[node]['class'] = i
                                nb[i] += 1
                                for neighbor in G_seeds.neighbors(seed):
                                    if G_seeds.node[neighbor]['class'] == -1:
                                        todo.add_node(neighbor)
                    i += 1
            poids = {}  # Used to ponder reccurent seeds
            for seed in ss:
                poids[seed] = ss.count(seed)

            # The harmonization
            for index in indexToScan:  # Proceding one connex part at a time
                # The central seed for each class is the one with the smallest
                # cummulated range to the others
                val = -1
                s = ss[0]
                for seed in G_seeds.nodes():
                    if classes[G_seeds.node[seed]['class']] == index:
                        # The seeds are computed using a kind of BFS :
                        # we consider all the neighbors of the visited set at
                        # one time. If there are seeds in them, they are added
                        # to the visited lot and we start again until we have
                        # all those of the current connex class. If not, all
                        # the neighbors are added to the visited lot at a time
                        # and we decrease the offset.
                        #
                        # So the offset determines how far from a connex seed
                        # class we search for other seeds that are to be
                        # associated with the current class.
                        # Thus, when finding a seed, the offset is reinitiated

                        todo = nx.Graph()
                        done = nx.Graph()
                        todo.add_node(seed)
                        done.add_node(seed)
                        found = 0
                        offset = Offset
                        dist = 0
                        total = 0  # The cumulated distance from the current
                        # seed to the other of the same class
                        while found < nb[classes[G_seeds.node[seed][
                                'class']]] + offset:
                            otherN = nx.Graph()  # Non seed neighbors
                            sourcesN = nx.Graph()  # seed neighbors
                            front = nx.Graph()  # The neighbors of otherN
                            foundAnOtherSource = False
                            for node in todo.nodes():
                                # The BFS part for a seed
                                if(G_seeds.has_node(node)):
                                    foundAnOtherSource = True
                                    total += dist * poids[node]
                                    found += 1
                                    offset = Offset
                                    if classes[G_seeds.node[seed][
                                                'class']] != \
                                            classes[G_seeds.node[node][
                                                'class']]:
                                        # We found a seed from a diffecent
                                        # class in the search range so we have
                                        # to merge the two classes
                                        nb[classes[G_seeds.node[seed][
                                                'class']]] += \
                                            nb[classes[G_seeds.node[node][
                                                'class']]]
                                        nb[classes[G_seeds.node[node][
                                                'class']]] = 0
                                        classes[G_seeds.node[node][
                                                'class']] = \
                                            classes[G_seeds.node[seed][
                                                'class']]
                                        if index == len(indexToScan)-1 \
                                            or index < len(indexToScan)-1 \
                                            and indexToScan[
                                                len(indexToScan)-1] != index:
                                            indexToScan.append(index)
                                            # the previous weights of the
                                            # current category have to be
                                            # computed again

                                    # Doing the BFS part
                                    for neighbor in G_i.neighbors(node):
                                        if not done.has_node(neighbor):
                                            sourcesN.add_node(neighbor)

                                # The BFS part for a not-seed node
                                elif not foundAnOtherSource:
                                    front.add_node(node)
                                    for neighbor in G_i.neighbors(node):
                                        if not done.has_node(neighbor):
                                            otherN.add_node(neighbor)

                            # Updating the visitated nodes in the BFS
                            if foundAnOtherSource:
                                done.add_nodes_from(sourcesN.nodes())
                                todo = sourcesN
                                todo.add_nodes_from(front.nodes())
                            else:
                                done.add_nodes_from(otherN.nodes())
                                todo = otherN
                            dist += 1
                            if found >= nb[classes[G_seeds.node[seed][
                                    'class']]]:
                                # We did not find any source so we got a step
                                # further from the considered connex class
                                offset -= 1
                        if val == -1 or total < val:
                            # We found a minimizing seed for the cummulated
                            # range factor in our connex class
                            val = total
                            s = seed
                # storing the minimizing seed for the connex class named index
                for z in range(0, index-len(sources)+1):
                    sources.append(s)
                sources[index] = s

            # The graph structure prevents duplicated sources in the output
            sol = nx.Graph()
            # We only keep the sources from the till existing connex classes
            for index in classes:
                sol.add_node(sources[index])
            return sol.nodes()
        else:
            # The harmonization was not planned
            return ss

    def run_once(G, G_i, prob):
        """
        In the first part, we compute the potential sources through generating
        an infection starting from the border of the infected subgraph : the
        infected nodes with clean neighbors.

        Then, a first, necessary, harmonisation is made to collapse the
        results of the infection ripple into single seeds through the same
        method as previously
        """
        Offset = 3
        frontier = nx.Graph()
        infected = nx.Graph()
        uninfected = nx.Graph(G_i)

        # Generating the frontier.
        # i_deg is used to access to the number of infected neighbors of
        # each node in the frontier in constant time
        for node in G_i:
            clear_neighbors = len(G.neighbors(node)) - \
                                len(G_i.neighbors(node))
            if clear_neighbors > 0:
                infected.add_node(node)
                uninfected.remove_node(node)
        for node in infected:
            for neighbor in G_i.neighbors(node):
                if not infected.has_node(neighbor):
                    if not frontier.has_node(neighbor):
                        frontier.add_node(neighbor, {'i_deg': 1})
                    else:
                        frontier.node[neighbor]['i_deg'] += 1

        ss = []
        old = []
        # Computing the ripple infection and getting the potential seeds.
        # A potential seed is a node where the ripple locally closes itself
        #       | | |
        #       v v v
        #   --> seeds <--
        #       ^ ^ ^
        #       | | |

        # We keep reccord of the two last results to deal with multiple seed
        # cases
        while len(infected.node) < len(G_i.node):
            old = ss
            ss = Custom.ripple_step(G_i, frontier, infected, uninfected, prob)

        for seed in old:
            ss.append(seed)

        # Harmonization as in the run function
        i = 0
        nb = []
        classes = []
        sources = []
        indexToScan = []
        G_seeds = nx.Graph()
        todo = nx.Graph()
        for seed in ss:
            G_seeds.add_node(seed, {'class': -1})

        for seed in G_seeds.nodes():
            for neighbor in G_i.neighbors(seed):
                if G_seeds.has_node(neighbor):
                    G_seeds.add_edge(seed, neighbor)

        for seed in G_seeds.nodes():
            if G_seeds.node[seed]['class'] != -1:
                continue
            else:
                G_seeds.node[seed]['class'] = i
                classes.append(i)
                indexToScan.append(i)
                nb.append(1)
                for neighbor in G_seeds.neighbors(seed):
                    todo.add_node(neighbor)
                while(len(todo.nodes()) > 0):
                    for node in todo.nodes():
                        todo.remove_node(node)
                        if G_seeds.node[node]['class'] == -1:
                            G_seeds.node[node]['class'] = i
                            nb[i] += 1
                            for neighbor in G_seeds.neighbors(seed):
                                if G_seeds.node[neighbor]['class'] == -1:
                                    todo.add_node(neighbor)
                i += 1

        for index in indexToScan:
            val = -1
            s = ss[0]
            for seed in G_seeds.nodes():
                if classes[G_seeds.node[seed]['class']] == index:
                    todo = nx.Graph()
                    done = nx.Graph()
                    todo.add_node(seed)
                    done.add_node(seed)
                    found = 0
                    offset = Offset
                    dist = 0
                    total = 0
                    while found < nb[classes[G_seeds.node[seed]['class']]] \
                            + offset:
                        otherN = nx.Graph()
                        sourcesN = nx.Graph()
                        front = nx.Graph()
                        foundAnOtherSource = False
                        for node in todo.nodes():
                            if(G_seeds.has_node(node)):
                                foundAnOtherSource = True
                                total += dist
                                found += 1
                                offset = Offset
                                if classes[G_seeds.node[seed]['class']] != \
                                        classes[G_seeds.node[node]['class']]:
                                    nb[classes[G_seeds.node[seed][
                                            'class']]] += \
                                        nb[classes[G_seeds.node[node][
                                            'class']]]
                                    nb[classes[G_seeds.node[node][
                                            'class']]] = 0
                                    classes[G_seeds.node[node][
                                                'class']] = \
                                        classes[G_seeds.node[seed][
                                                'class']]
                                    if index == len(indexToScan)-1 \
                                        or index < len(indexToScan)-1 \
                                        and indexToScan[
                                            len(indexToScan)-1] != index:
                                        indexToScan.append(index)

                                for neighbor in G_i.neighbors(node):
                                    if not done.has_node(neighbor):
                                        sourcesN.add_node(neighbor)

                            else:
                                front.add_node(node)
                                for neighbor in G_i.neighbors(node):
                                    if not done.has_node(neighbor):
                                        otherN.add_node(neighbor)
                        if foundAnOtherSource:
                            done.add_nodes_from(sourcesN.nodes())
                            todo = sourcesN
                            todo.add_nodes_from(front.nodes())
                        else:
                            done.add_nodes_from(otherN.nodes())
                            todo = otherN
                        dist += 1
                        if found >= nb[classes[G_seeds.node[seed]['class']]]:
                            offset -= 1
                    if val == -1 or total < val:
                        val = total
                        s = seed
            for z in range(0, index-len(sources)+1):
                sources.append(s)
            sources[index] = s

        sol = nx.Graph()
        for index in classes:
            sol.add_node(sources[index])

        return sol.nodes()

    def ripple_step(G_i, frontier, infected, uninfected, prob):
        """
        Generating a step in the infection ripple simulation
        using a randomized BFS like algorithm
        """

        seeds = []

        # List containing the nodes from the frontier at time t with k
        # infected neighbors
        frontier_degree_t = []

        # List containing the newly infected nodes
        infected_t = []

        # Generating the lists containing the nodes with i infected neighbors
        for node in frontier:
            i = frontier.node[node]['i_deg']
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
                if prob < 1:
                    f_j = len(frontier_degree_t[j])
                    p_j = 1-(1-prob)**(j+1)
                    n_j = int(min(np.floor(p_j*(f_j+1)), f_j))
                    # The f_j/prob implicates n_j > f_j
                    infected_t.append(np.random.choice(frontier_degree_t[j],
                                                       n_j, replace=False))
                else:
                    infected_t.append(frontier_degree_t[j])

        # Updating the frontier
        for j in infected_t:
            for node in j:
                infected.add_node(node)
                frontier.remove_node(node)
                seeds.append(node)
        ss = []
        for seed in seeds:
            if len(uninfected.neighbors(seed)) > 0:
                for neighbor in uninfected.neighbors(seed):
                    if not infected.has_node(neighbor):
                        if not frontier.has_node(neighbor):
                            frontier.add_node(neighbor, {'i_deg': 1})
                        else:
                                frontier.node[neighbor]['i_deg'] += 1
            else:
                ss.append(seed)
        for seed in seeds:
            uninfected.remove_node(seed)

        return seeds  # ou ss
