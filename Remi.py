import sys
import networkx as nx
from sets import Set
from scipy.stats import bernoulli

p = 0.65;
n_simul = 15

class RemiAlgorithm:
	def __init__(self):
		p = 0.47;
		n_simul = 5

	# generates the infection for one turn
	def turn(self,G,source,non_infected):
		new_infected = Set()
		for s in non_infected:
			# check how many of its neighbours are infected
			neighbours = nx.all_neighbors(G,s)
			N = 0
			for k in neighbours:
				N = N+1
				if k in non_infected:
					N = N-1
			# the probability of being infected depends on the number of infected neighbours
			p_s = 1 - ((1-p)**N)
			if(bernoulli.rvs(p,0)==1):
				new_infected.add(s)
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
			non_infected = self.turn(G,source,non_infected)
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
		#print(source)
		return source
