#!/usr/bin/env python
# coding: utf-8

# In[3]:


import xml.etree.ElementTree as ET
import networkx as nx
from networkx.algorithms.similarity import graph_edit_distance
import os

def parse_bpmn_file(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    bpmn_elements = []
    for element in root.findall('.//bpmn:process/bpmn:*', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}):
        element_id = element.get('id')
        element_type = element.tag.split('}')[-1]  
        
        if element_id:  
            next_elements = [outgoing.get('targetRef') for outgoing in element.findall('.//bpmn:outgoing', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'})]
            
            bpmn_elements.append({
                'id': element_id,
                'type': element_type,
                'next': next_elements
            })
    
    return bpmn_elements

def create_bpmn_graph(bpmn_elements):
    G = nx.DiGraph()
    for element in bpmn_elements:
        element_id = element['id']
        if element_id:  
            G.add_node(element_id, type=element['type'])
            for next_element in element.get('next', []):
                if next_element:  
                    G.add_edge(element_id, next_element)
    return G

def calculate_structural_similarity(bpmn1, bpmn2):
    graph1 = create_bpmn_graph(bpmn1)
    graph2 = create_bpmn_graph(bpmn2)
    
    distance = graph_edit_distance(graph1, graph2)
    max_distance = max(len(graph1.nodes), len(graph2.nodes))
    similarity = 1 - (distance / max_distance) if max_distance > 0 else 1.0
    
    return similarity

file_bpmn1 = 'diagram (5).bpmn'  
file_bpmn2 = 'diagram (6).bpmn'  


bpmn1 = parse_bpmn_file(file_bpmn1)
bpmn2 = parse_bpmn_file(file_bpmn2)


similarity = calculate_structural_similarity(bpmn1, bpmn2)
print(f"Structural similarity between bpmn is: {similarity:.2f}")





