#!/usr/bin/env python
# coding: utf-8

# In[9]:





# In[18]:

#https://processintelligence.solutions/static/api/2.7.11/api.html#conversion-pm4py-convert
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
from pm4py.algo.conversion.petri_net_to_networkx import converter as petri_to_nx
import networkx as nx
import matplotlib.pyplot as plt

# Path to the CSV file
path = 'Framework-data1.csv'  # Enter path to the csv file

# Load the CSV data
data = pd.read_csv(path)

# Rename the columns to fit the event log format
cols = ['case:concept:name', 'concept:name', 'time:timestamp']
data.columns = cols

# Convert 'time:timestamp' to datetime format and 'concept:name' to string
data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
data['concept:name'] = data['concept:name'].astype(str)

# Convert the DataFrame to an event log
expected_log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)

# Apply the Alpha Miner algorithm to discover the Petri net
petri_net, initial_marking, final_marking = alpha_miner.apply(expected_log)

# Check the soundness of the Petri net
is_sound, diagnostics = pm4py.algo.conformance.petri_net.soundness_checker.apply(petri_net, initial_marking, final_marking)

# Print the results
print(f"Is the Petri net sound? {is_sound}")
print("Diagnostics:", diagnostics)

# Convert Petri Net to NetworkX graph and visualize
nx_graph = petri_to_nx.convert_petri_net_to_networkx(petri_net, initial_marking, final_marking)
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color="lightblue", node_size=2000, font_size=10, font_weight='bold')
plt.show()


# In[ ]:




