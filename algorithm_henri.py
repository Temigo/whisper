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

    def run(G, G_i, prob):
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
        seeds = []
        ss = []
        old = []
        while len(infected.node) < len(G_i.node):
            old = ss
            ss = Custom.ripple_step(G_i, frontier, infected, uninfected, prob)

#            for seed in ss:
#                G_seeds.add_node(seed, {'part':-1})
#            for seed in  ss:
#                for neighbors in G_i.neighbors(seed):
#                    if G_seeds.has_node(neighbor):
#                        G_seeds.add_edge(seed, neighbor)
#
#
#            for seed in ss:
#                seeds.append(seed)
            seeds = ss
        
        for seed in old:
            ss.append(seed)
        # Parcours en profondeur pour détecter les classes d'équivalences
        i = 0
        nb = []
        classes = []
        sources = []
        indexToScan = []
        G_seeds = nx.Graph()
        todo = nx.Graph()
        for seed in ss:
            G_seeds.add_node(seed, {'class': -1})
            # edging
        for seed in G_seeds.nodes():
            for neighbor in G_i.neighbors(seed):
                if G_seeds.has_node(neighbor):
                    G_seeds.add_edge(seed, neighbor)
        # Les classes de connexité :
        for seed in G_seeds.nodes():
#            print("Init : seed "+str(seed)+" cat : "+str(G_seeds.node[seed]['class']))
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
#            print("     cat finale : "+str(G_seeds.node[seed]['class']))
        # Le parcours en profondeur pour sélectionner la source centrale
        # dans chaque classe de connexité

#        print(indexToScan)
#        print(nb)
#        print(G_seeds.nodes(data = True))
        for index in indexToScan:
#            print("en cours : "+str(index))
            val = -1
            s = ss[0]
            for seed in G_seeds.nodes():
#                print("     source : "+str(seed))
#                print("     cat : "+str(classes[G_seeds.node[seed]['class']]))
#                print("     nbr : "+str(nb[classes[G_seeds.node[seed]['class']]]))
                if classes[G_seeds.node[seed]['class']] == index:
                    todo = nx.Graph()
                    done = nx.Graph()
                    todo.add_node(seed)
                    done.add_node(seed)
                    found = 0
                    offset = 2
                    dist = 0
                    total = 0
#                    print("          Famille : "+str(classes[G_seeds.node[seed]['class']]))
#                    print("     nb : "+str(nb[classes[G_seeds.node[seed]['class']]]))
                    while found < nb[classes[G_seeds.node[seed]['class']]] \
                            + offset:
                        nextN = nx.Graph()
                        for node in todo.nodes():
                            if(G_seeds.has_node(node)):
                                total += dist
                                found += 1
#                                print("Trouvés : "+str(found))
#                                print("Cible : "+str(nb[classes[G_seeds.node[seed]['class']]] \
#                            + offset))
                                offset = 2
#                                print("               offset init : "+str(node)+" de classe : "+str(classes[G_seeds.node[node]['class']] \
#                                ) +" trouvés :"+str(found)+"/"+str(nb[classes[G_seeds.node[seed]['class']]]+ offset))
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
#                                print("          voisins : "+str(neighbor))
                                if not done.has_node(neighbor):
#                                    print(str(neighbor)+" déjà")
                                    nextN.add_node(neighbor)
                                    done.add_node(neighbor)
#                        print(done.nodes())
#                        print("next : "+str(nextN.nodes()))
                        dist += 1
                        todo = nextN
                        if found >= nb[classes[G_seeds.node[seed]['class']]]:
                            offset -= 1
#                            print("                offset : "+str(offset))
                    if val == -1 or total < val:
                        val = total
                        s = seed
#            print("tentative de définir la source pou l'index : "+str(index))
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
