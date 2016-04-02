# Tests for algorithm Remi

from Remi import RemiAlgorithm
import networkx as nx
import unittest

class TestRemi(unittest.TestCase):
	# Methode appelee avant chaque test
	def setUp(self):
		pass
		
	# Methode appelee apres chaque test
	def tearDown(self):
		pass
				
	def test_elementary(self):
		g = nx.Graph()
		g.add_nodes_from([1, 2, 3, 4, 5])
		g.add_edges_from([(1, 2), (2, 1), (1, 3), (3, 1), (2, 4), (4, 2), (5, 2), (2, 5)])
		remi = RemiAlgorithm()
		remi.run(g)
