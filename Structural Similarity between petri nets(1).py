

import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import graph_edit_distance

# Define Petri Net 1
G1 = nx.DiGraph()
# Adding places and transitions
places = ["P1", "P2", "P3"]
transitions = ["T1", "T2"]
G1.add_nodes_from(places, bipartite=0)
G1.add_nodes_from(transitions, bipartite=1)

# Adding arcs
arcs = [("P1", "T1"), ("T1", "P2"), ("P2", "T2"), ("T2", "P3")]
G1.add_edges_from(arcs)

# Define Petri Net 2
G2 = nx.DiGraph()
# Adding places and transitions
places = ["P1", "P2", "P3", "P4"]
transitions = ["T1", "T2"]
G2.add_nodes_from(places, bipartite=0)
G2.add_nodes_from(transitions, bipartite=1)

# Adding arcs
arcs = [("P1", "T1"), ("T1", "P2"), ("P2", "T2"), ("T2", "P4")]
G2.add_edges_from(arcs)

# Visualize Petri Net 1
plt.figure(figsize=(12, 6))
plt.subplot(121)
pos = nx.spring_layout(G1, k=0.9)
nx.draw(G1, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=12, node_shape='o')
plt.title("Petri Net 1")

# Visualize Petri Net 2
plt.subplot(122)
pos = nx.spring_layout(G2, k=0.9)
nx.draw(G2, pos, with_labels=True, node_size=1000, node_color="lightgreen", font_size=12, node_shape='o')
plt.title("Petri Net 2")

plt.tight_layout()
plt.show()

# Calculate Graph Edit Distance
ged = graph_edit_distance(G1, G2)

# Normalize GED to a similarity measure
max_possible_ged = max(len(G1.nodes()), len(G2.nodes()))
similarity = 1 - (ged / max_possible_ged)  # Normalizing GED to a similarity measure

print("Structural Similarity (GED):", similarity)
