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
print(f"La similarità strutturale tra i due BPMN è: {similarity:.2f}")


# In[1]:


import xml.etree.ElementTree as ET
import networkx as nx

def parse_bpmn_with_types(file_name):
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    # Lista per tenere traccia degli elementi BPMN
    bpmn_elements = []
    
    # Cerchiamo tutti gli elementi BPMN nel processo
    for element in root.findall('.//bpmn:process/bpmn:*', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}):
        element_id = element.get('id')
        element_type = element.tag.split('}')[-1]  # Prendiamo il tipo del nodo
        
        if element_id:  # Assicuriamoci che l'elemento abbia un ID valido
            next_elements = [outgoing.get('targetRef') for outgoing in element.findall('.//bpmn:outgoing', namespaces={'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'})]
            
            bpmn_elements.append({
                'id': element_id,
                'type': element_type,  # Includiamo il tipo di elemento
                'next': next_elements  # Colleghiamo i nodi successivi
            })
    
    return bpmn_elements

def create_weighted_bpmn_graph(bpmn_elements):
    G = nx.DiGraph()
    
    # Mappatura di pesi basati sui tipi di nodi (più complessi possono avere pesi maggiori)
    type_weights = {
        'startEvent': 1,
        'endEvent': 1,
        'task': 2,
        'exclusiveGateway': 3,
        'parallelGateway': 3
        # Puoi aggiungere altri tipi e pesi in base alle necessità
    }
    
    for element in bpmn_elements:
        element_id = element['id']
        element_type = element['type']
        node_weight = type_weights.get(element_type, 1)  # Assegniamo un peso predefinito di 1 se il tipo non è mappato
        
        if element_id:  
            G.add_node(element_id, type=element_type, weight=node_weight)  # Aggiungiamo nodi con il loro peso
            for next_element in element.get('next', []):
                if next_element:  
                    G.add_edge(element_id, next_element)  # Colleghiamo gli archi tra i nodi
    
    return G

def calculate_weighted_similarity(bpmn1, bpmn2):
    # Creiamo i grafi pesati per i due BPMN
    graph1 = create_weighted_bpmn_graph(bpmn1)
    graph2 = create_weighted_bpmn_graph(bpmn2)
    
    # Funzione di similarità tra i due grafi
    def node_match(n1, n2):
        return n1['type'] == n2['type']  # Confrontiamo i nodi in base al loro tipo
    
    # Calcolo della distanza di edit pesata
    distance = nx.graph_edit_distance(graph1, graph2, node_match=node_match)
    
    # Calcolo del massimo numero di nodi
    max_nodes = max(len(graph1.nodes), len(graph2.nodes))
    
    # Similarità pesata: 1 - (distanza edit pesata / max_nodes)
    similarity = 1 - (distance / max_nodes) if max_nodes > 0 else 1.0
    
    return similarity

# Caricamento dei file BPMN
file_bpmn1 = 'diagram (5).bpmn'
file_bpmn2 = 'diagram (6).bpmn'

# Parsing dei file BPMN con inclusione dei tipi di nodo
bpmn1 = parse_bpmn_with_types(file_bpmn1)
bpmn2 = parse_bpmn_with_types(file_bpmn2)

# Calcolo della similarità pesata
similarity = calculate_weighted_similarity(bpmn1, bpmn2)
print(f"La similarità strutturale pesata tra i due BPMN è: {similarity:.2f}")


# In[ ]:




