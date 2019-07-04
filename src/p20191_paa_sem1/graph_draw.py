import itertools
import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def drawGraphWithCommunities(graph: nx.Graph, communities: Dict[int, List[int]] = None, weighted=False):
    if communities is None:
        communities = {n: [vertex] for n, vertex in enumerate(graph)}

    colors = [
        '#F44336',
        '#E91E63',
        '#9C27B0',
        '#673AB7',
        '#3F51B5',
        '#2196F3',
        '#03A9F4',
        '#00BCD4',
        '#009688',
        '#4CAF50',
        '#8BC34A',
        '#CDDC39',
        '#FFEB3B',
        '#FFC107',
        '#FF9800',
        '#FF5722',
        '#795548',
        '#9E9E9E',
        '#607D8B'
    ]
    colors = random.sample(colors, len(colors))

    from random import seed as pythonSeed
    from numpy.random import seed as numpySeed
    for seed in (pythonSeed, numpySeed):
        seed(42)

    layout = nx.spring_layout(graph)

    plt.figure(figsize=(32, 16))
    for community, vertexes in communities.items():
        color = colors[community]
        nx.draw_networkx_nodes(graph, layout, vertexes, node_color=color, node_size=12*10**3)

    nx.draw_networkx_edges(graph, layout, width=10)
    if weighted:
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, layout, font_size=48, font_color='black', font_family='hack', edge_labels=labels)
    nx.draw_networkx_labels(graph, layout, font_size=64, font_color='white', font_family='hack')
