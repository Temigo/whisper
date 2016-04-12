# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 14:27:14 2015

@author: h

le paramètre j permet de gagner en constance et précision dans l'exécution du code au détriment d'un peu de vitesse
A tester si c'est vraiment efficace
"""

import networkx as nx
import numpy as np


class Custom:
    def __init__(self):
        pass

    def run(G, G_i, prob, j=1):
        ss = []
        Offset = 3
        for i in range(0, j):
            for seed in Custom.run_once(G, G_i, prob):
                ss.append(seed)
        if j > 1:
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
                                        nb[classes[G_seeds.node[seed]['class']]] += \
                                                nb[classes[G_seeds.node[node]['class']]]
                                        nb[classes[G_seeds.node[node]['class']]] = 0
                                        classes[G_seeds.node[node]['class']] = \
                                                classes[G_seeds.node[seed]['class']]
                                        if index == len(indexToScan)-1 or \
        index < len(indexToScan)-1 and indexToScan[len(indexToScan)-1] != index:
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
                        if val == -1 or total/ss.count(seed) < val:
                            val = total/ss.count(seed)
                            s = seed
                for z in range(0, index-len(sources)+1):
                    sources.append(s)
                sources[index] = s
    
            sol = nx.Graph()
            for index in classes:
                sol.add_node(sources[index])
            return sol.nodes()   
        else:     
            return ss

    def run_once(G, G_i, prob):
        Offset = 3
        frontier = nx.Graph()
        infected = nx.Graph()
        uninfected = nx.Graph(G_i)

        for node in G_i:
            clear_neighbors = len(G.neighbors(node)) -\
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
        while len(infected.node) < len(G_i.node):
            old = ss
            ss = Custom.ripple_step(G_i, frontier, infected, uninfected, prob)

        for seed in old:
            ss.append(seed)

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
                                    nb[classes[G_seeds.node[seed]['class']]] += \
                                            nb[classes[G_seeds.node[node]['class']]]
                                    nb[classes[G_seeds.node[node]['class']]] = 0
                                    classes[G_seeds.node[node]['class']] = \
                                            classes[G_seeds.node[seed]['class']]
                                    if index == len(indexToScan)-1 or \
    index < len(indexToScan)-1 and indexToScan[len(indexToScan)-1] != index:
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
        """ Generating a step in the infection simulation"""

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
