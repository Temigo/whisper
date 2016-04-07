import sys
import networkx as nx
from sets import Set
from scipy.stats import bernoulli
import math

p = 0.47;
n_simul = 15

class RemiAlgorithmOriginal:
	def __init__(self):
		p = 0.47;
		n_simul = 5

	# calculates the maximal N = E_i/P_i such that
	# for all j, P_j*N <= |E_j|
	def max_N(self,E):
		for E_i in E:
			if(E_i != None):
				P_i = 1 - ((1-p)**len(E_i))
				N = math.floor(len(E_i)/P_i)
				# condition: for all j, P_j*N <= |E_j|
				condition = True
				for E_j in E:
					if(E_j != None):
						P_j = 1 - ((1-p)**len(E_j))
						if(P_j*N > len(E_j)):
							condition = False
				if(condition == True):
					return N
		return 0

	# generates the set with the new infected nodes
	def new_infected_set(self,E,N):
		new_infected = Set()
		# for each E[i], pick N[i] = N*P[i] nodes to be infected
		for E_i in E:
			if(E_i != None):
				P_i = 1 - ((1-p)**len(E_i))
				N_i = math.ceil(N*P_i)
				for k in range(0,int(N_i)):
					if(len(E_i) > 0):
						new_infected.add(E_i.pop())
		return new_infected

	# generates the vector E with the infected sets:
	# # i represents the number of infected neighbours
	# E[i] is the list with i infected neighbours
	def infected_sets(self,G,non_infected):
		E = [None]*G.number_of_nodes()
		for s in non_infected:
			# check how many of s neighbours are infected
			neighbours = nx.all_neighbors(G,s)
			n = 0
			for k in neighbours:
				n = n+1
				if k in non_infected:
					n = n-1
			if(E[n] == None):
				E[n] = Set()
			E[n].add(s)
		return E

	# generates the infection for one turn
	def turn(self,G,non_infected):
		E = self.infected_sets(G,non_infected)
		N = self.max_N(E)
		new_infected = self.new_infected_set(E,N)
		# remove them from the non infected set
		non_infected = non_infected.difference(new_infected)
		return non_infected

	# calculates the time for a simulation
	def time_infection(self,G,source):
		# (initial time of the infection)
		t = 0 
		non_infected = Set()
		# insert everyone but the source
		for s in G:
			non_infected.add(s)
		non_infected.remove(source)
		# propagate the infection
		while(len(non_infected) > 0):
			non_infected = self.turn(G,non_infected)
			t = t+1
		return t

	# calculates the mean of the infection time for a node s
	def mean(self,G,source):
		sum = 0.0
		for i in range(0, n_simul):
			sum = sum + self.time_infection(G,source)
		return sum/n_simul

	# choose the node with the smallest approximated mean of the infection time
	# G must be the infected graph
	def run(self, G):
		source = None
		min_M = sys.maxint
		for s in G:
			M = self.mean(G,s)
			#print(str(s)+": "+str(M)) # prints "node number: mean of infection times"
			if (M < min_M):
				source = s
				min_M = M
		# print(source)
		return source
