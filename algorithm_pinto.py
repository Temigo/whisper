# -*- coding: utf-8 -*-
# Algorithms used for rumor source inference in a graph
# - Pinto algorithm

import networkx as nx # on ne va pas utiliser cette biblioteque la?
# from graph import Node, Edge, Graph

class GraphPinto:
	def __init__(self):
		pass
	# G is the graph
	# O is a vector with the observer nodes
	
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
	# path: i = number of edges between the node and the supposed source (SEE IF USEFULL)
	#
	# time: t = time when the observer receives the information
	#
	# Example: G.add_node(1,informed=0,observer=0,path=0,time=0)
	#####################################################################

	def Algorithm(G,O):

        ###### CALCULATE ARRAY D (observed delay) ######
		for i in range(0,len(O)-1)
			d[i] = O[i+1].time - O[i].time

		# calculates S for the first node: fulfills max
		MAX = FUNCTION(first_node,d,Tbfs)
		source = first_node
				
		# calculates the maximum S
		for s in G
			# constructs a T = Tbfs ####### DESCOBRIR COMO FAZ
			# calculates F = FUNCTION(s,d,T) ######## DESCOBRIR COMO FAZ
			# if F > MAX:
				# MAX = F
				# source = s

	# Function S to be calculated
	def FUNCTION(s,d,T):
		# calculates mi ############ DESCOBRIR COMO FAZ
		# calculates delta ############# DESCOBRIR COMO FAZ
		# return mi(transpose)*delta^(-1)*(d-0.5mi) ######### DESCOBRIR COMO FAZ
		pass

	# Function to construct the breath-first-search
	def graph_to_bfs(G):
		pass
