#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import matplotlib.pyplot as plt

# Define the original graph G
G1 = nx.DiGraph()
nodes_sequence = ["Task1", "Task2", "Task3", "Task4", "Task5"]
G1.add_nodes_from(nodes_sequence)
edges = [("Task1", "Task2"), ("Task2", "Task3"), ("Task3", "Task4")]
G1.add_edges_from(edges)

#

# Aggiungi il nodo del gateway e i flussi
gateway_node = "OR"
incoming_flows = ["Task1"]
outgoing_flows = ["Task2", "Task3"]
G1.add_node(gateway_node)
for source_node in incoming_flows:
    G1.add_edge(source_node, gateway_node)
for target_node in outgoing_flows:
    G1.add_edge(gateway_node, target_node)

# Visualizzazione del grafo
plt.figure(figsize=(15, 10))
plt.subplot(121)
pos = nx.spring_layout(G1, k=0.9)
nx.draw(G1, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=12)
plt.title("Structural Graph 1 with Gateway")
plt.tight_layout()
plt.show()

# Visualize the total graph
plt.figure(figsize=(16, 8))


# In[2]:


import networkx as nx
import matplotlib.pyplot as plt

# Define the original graph G2
G2 = nx.DiGraph()
nodes_sequence = ["Task1", "Task2", "Task3", "Task4", "Task51"]
G2.add_nodes_from(nodes_sequence)
edges = [("Task1", "Task2"), ("Task2", "Task3"), ("Task3", "Task4"),
         ( "Task4","Task51")]
G2.add_edges_from(edges)

# Aggiungi il nodo del gateway e i flussi
gateway_node = "XOR"
incoming_flows = ["Task1"]
outgoing_flows = ["Task2", "Task3"]
G2.add_node(gateway_node)
for source_node in incoming_flows:
    G2.add_edge(source_node, gateway_node)
for target_node in outgoing_flows:
    G2.add_edge(gateway_node, target_node)

# Visualizzazione del grafo
plt.figure(figsize=(15, 10))
plt.subplot(121)
pos = nx.spring_layout(G2, k=0.9)
nx.draw(G2, pos, with_labels=True, node_size=1000, node_color="lightgreen", font_size=12)
plt.title("Structural Graph 2 with Gateway")
plt.tight_layout()
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance

ged = graph_edit_distance(G1, G2)

max_possible_ged = max(len(G1), len(G2))
similarity = 1 - (ged / max_possible_ged)  # Normalizing GED to a similarity measure


print("Structural Similarity (GED):", similarity)


# In[4]:


import networkx as nx
import matplotlib.pyplot as plt



# Calculate SEM similarity
intersection_size = len(set(G1.nodes()).intersection(set(G2.nodes())))

synonym_penalty = 0
for s in set(G1.nodes()).difference(set(G2.nodes())):
    for l in set(G2.nodes()).difference(set(G1.nodes())):
        if s == l:  # Checking for synonymous pairs
            synonym_penalty += 1

max_set_size = max(len(G1.nodes()), len(G2.nodes()))

sem_similarity = 1.0 * intersection_size + 0.75 * synonym_penalty
max_similarity = max(len(G1.nodes()), len(G2.nodes()))

normalized_sem_similarity = sem_similarity / max_similarity


print("Normalized SEM Similarity:", normalized_sem_similarity)


# In[1]:


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


# In[ ]:




