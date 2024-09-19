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


