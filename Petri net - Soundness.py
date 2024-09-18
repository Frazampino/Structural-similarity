#!/usr/bin/env python
# coding: utf-8

# In[9]:





# In[18]:


import pandas as pd
import pm4py  # Import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner

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
is_sound, diagnostics = pm4py.analysis.check_soundness(petri_net, initial_marking, final_marking, print_diagnostics=True)

# Print the results
print(f"Is the Petri net sound? {is_sound}")
print("Diagnostics:", diagnostics)


# In[ ]:




