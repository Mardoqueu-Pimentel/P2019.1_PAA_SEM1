from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Set, TypeVar

import networkx as nx
import numpy as np

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def reverseDict(d: Dict[T1, T2]) -> Dict[T2, Set[T1]]:
	newDict = {}
	for k, v in d.items():
		if v not in newDict:
			newDict[v] = set()
		newDict[v].add(k)
	return newDict


def nodeCommunitiesToPartition(nodeCommunities: dict):
	communities = reverseDict(nodeCommunities)
	return list(communities.values())


def modularity(graph: nx.Graph, partition: List[Set[int]], weight='weight') -> float:
	"""
	Calculates the modularity of a graph.

	Parameters
	----------
	graph: nx.Graph
			A NetworkX Graph.

	partition: List[Set[int]]
			A list of sets of ints, where each set represents a community and each
			int represents a vertex

	weight: str
			The name of the key used in the node to represent it's weight

	Returns
	-------
	Q: float
			The modularity

	Examples
	________
	>>> g = nx.complete_graph(5)
	>>> q = modularity(g, [{0, 1, 2}, {3, 4}])
	-0.12000000000000002

	"""
	m = graph.size(weight=weight)
	norm = 1/(2*m)
	vertexWeightedDegrees = dict(graph.degree(weight=weight))

	def val(i, j):
		edgeWeight = graph[i][j].get(weight, 1) if j in graph[i] else 0
		w = edgeWeight*2 if i == j else edgeWeight
		return w-(vertexWeightedDegrees[i]*vertexWeightedDegrees[j]*norm)

	return sum(val(i, j) for com in partition for i, j in product(com, repeat=2))*norm


def deltaModularity(graph: nx.Graph, nodeCommunities: dict, vertex: int, community: int, weight='weight') -> float:
	communities = reverseDict(nodeCommunities)
	m = graph.size(weight=weight)

	sEdgesIC = sumOfEdgesBetweenVertexAndCommunity(graph, vertex, communities[community], weight)
	sEdgesXC = sumOfEdgesBetweenCommunityAndGraph(graph, communities[community], weight)
	sEdgesIX = sumOfEdgesBetweenVertexAndGraph(graph, vertex, weight)

	return (sEdgesIC/(2*m))-((sEdgesXC*sEdgesIX)/(2*(m**2)))


def sumOfEdgesBetweenVertexAndCommunity(graph: nx.Graph, vertex: int, community: set, weight='weight') -> int:
	return sum(
		neighborInfo.get(weight, 1)
		for neighbor, neighborInfo in graph[vertex].items()
		if neighbor in community
	)


def sumOfEdgesBetweenCommunityAndGraph(graph: nx.Graph, community: Set[int], weight='weight') -> int:
	return sum(
		sum(
			info.get(weight, 1)
			for neighbor, info in graph[vertex].items()
			if neighbor not in community
		)
		for vertex in community
	)


def sumOfEdgesBetweenVertexAndGraph(graph: nx.Graph, vertex: int, weight='weight') -> int:
	return graph.degree(vertex, weight=weight)


def adjMToGraph(adjM: List[List[int]], weighted=False) -> nx.Graph:
	graph = nx.Graph()
	adjM = np.array(adjM)

	rows, cols = np.where(adjM > 0)
	if weighted:
		graph.add_weighted_edges_from(((row, col, adjM[row][col]) for row, col in zip(rows, cols)))
	else:
		graph.add_edges_from(zip(rows, cols))

	return graph


@dataclass(unsafe_hash=True)
class GraphInfo(object):
	m = None
	norm = None
	vertexesDegrees = None
	vertexesCommunities = None
	communitiesVertexes = None

	communitiesWeightedDegreeSum = None
	communitiesEdgeSum = None

	def __init__(self, graph: nx.Graph, vertexesCommunities=None, weight='weight'):
		self.m = graph.size(weight=weight)
		self.ma = 1 / self.m
		self.mb = 1 / (2 * self.m)
		self.vertexesDegrees = graph.degree(weight=weight)
		self.vertexesCommunities = vertexesCommunities if vertexesCommunities else {}
		self.communitiesVertexes = reverseDict(self.vertexesCommunities)
		self.communitiesWeightedDegreeSum = defaultdict(float)
		self.communitiesEdgeSum = defaultdict(float)

		for i in graph:
			com = vertexesCommunities[i]
			self.communitiesWeightedDegreeSum[com] += self.vertexesDegrees[i]
			for j, edge in graph[i].items():
				w = edge.get(weight, 1) if i == j else edge.get(weight, 1)/2
				self.communitiesEdgeSum[com] += w

	def modularity(self):
		def communityModularity(com):
			a = self.communitiesEdgeSum[com] * self.ma
			b = self.communitiesWeightedDegreeSum[com] * self.mb
			return a - b**2

		return sum(communityModularity(com) for com in self.communitiesVertexes.values())


def oneLevel(graph: nx.Graph, info: GraphInfo, resolution):
	modified = True


def louvainMethod(graph: nx.Graph, weight='weight'):
	currGraph = graph.copy()

	info = GraphInfo(currGraph, weight)
	infoList = []
