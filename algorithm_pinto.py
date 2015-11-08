# Algorithms used for rumor source inference in a graph
# - Pinto algorithm

import networkx as nx
from numpy import matrix
from numpy import linalg
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

class Node(object):
    def __init__(self):
        self.name = ""
        self.informed = 0
        self.observer = 0
        self.path = 0
        self.time = 0
        self.bfs = False
        self.children = []

# O is a vector with the observer nodes
# propagation delays are RVs with Gaussian distribution N(mi,sigma2)
class GraphPinto:
    def __init__(self):
        pass

    def Algorithm(G,O,mi,sigma2):
        d = observed_delay(O)
        # calculates F for the first node: fulfills max
        MAX = MAIN_FUNCTION(first_node,d,Tbfs,mi)
        source = first_node # SEE HOW WE CAN DO IT                          
        # calculates the maximum F
        for s in G:
            T = graph_to_bfs(G,s)
            F = MAIN_FUNCTION(s,d,T,mi)
            if F > MAX:
                MAX = F
                source = s
        return source

    # MAIN_FUNCTION S to be calculated
    def MAIN_FUNCTION(s,d,T,mi):
        mi_s = deterministic_delay(s,O,mi)
        delta = delay_covariance(O,sigma)
        return mi.T*(delta.I)*(d-(0.5*mi))

    # constructs the breath-first-search tree for a root s
    def graph_to_bfs(G,s):
        s.bfs = True
        stack = s
        while stack:
            vertex = stack.pop(0)
            voisinage = G.neighbors(vertex)
            for (i = 0 to len(voisinage)-1):
                if(!voisinage[i].bfs):
                    voisinage[i].bfs = True
                    stack.append(voisinage[i])
                    vertex.children.append(voisinage[i])
        # reset the variables bfs
        for v in G:
            v.bfs = False
        return s

    # calculates array d (observed delay)
    def observed_delay(O):
        for i in range(0,len(O)-1):
            d[i] = O[i+1].time - O[i].time
        return d

    # calculates array mi_s (deterministic delay)
    def deterministic_delay(s,O,mi):
        constant = height_node(s,O[0])
        for(i in range(0, len(O)-1):
            mi_s[i] = height_node(s,O[i+1])-constant
        mi_s = mi*mi_s
        return mi_s

    # calculates the height of a node in the tree T (recursive)
    def height_node(T,node):
        if(len(T.children) == 0):
            return 0
        if(node in T.children):
            return 1
        H = 0
        for(i in range(0, len(T.children))):
            H = H+height_node(T.children[i],node)
        if(H != 0):
            return H+1
        return 0

    # calculates the array delta (delay covariance)
    def delay_covariance(O,sigma):
        for(k in range(0, len(O)-1)):
            for(i in range(0, len(O)-1)):
                if(i == k):
                    delta[k][i] = height_node(O[0],O[k+1])
                else:
                    delta[k][i] =  math.fabs(height_node(O[0],O[k+1])-height_node(O[0],O[i+1]))
        delta = delta*(sigma**2)
        return delta
