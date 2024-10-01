#!/usr/bin/env python
# coding: utf-8

# In[4]:


import xml.etree.ElementTree as ET
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Funzione per estrarre le attività da un file BPMN
def extract_tasks_from_bpmn(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    return [t for t in tasks if t]  # Ritorna solo le attività non vuote


def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r'\W+', ' ', text)  
    return text


def calculate_cosine_similarity(process_a_tasks, process_b_tasks):
    # Preprocessamento delle etichette
    process_a_tasks = [preprocess_text(task) for task in process_a_tasks]
    process_b_tasks = [preprocess_text(task) for task in process_b_tasks]

    
    all_tasks = process_a_tasks + process_b_tasks

  
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tasks)
    
    
    process_a_matrix = tfidf_matrix[:len(process_a_tasks)]
    process_b_matrix = tfidf_matrix[len(process_a_tasks):]

   
    similarity_matrix = cosine_similarity(process_a_matrix, process_b_matrix)
    
    return similarity_matrix


def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

def main(bpmn_file_a, bpmn_file_b):
     process_a_tasks = extract_tasks_from_bpmn(bpmn_file_a)
    process_b_tasks = extract_tasks_from_bpmn(bpmn_file_b)

   
    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks)
    
    
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
   
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")

# example
bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'

main(bpmn_file_a, bpmn_file_b)

#gatways and tasks comparison
# In[6]:
import xml.etree.ElementTree as ET
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def extract_elements_from_bpmn(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei nomi delle attività dai nodi <bpmn:task>
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    
    
    gateways = []
    gateway_types = ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway']
    
    for g_type in gateway_types:
        gateways += [g_type + " " + (elem.get('name') if elem.get('name') else '') 
                     for elem in root.findall(f'.//bpmn:{g_type}', namespace)]
    
    
    elements = [t for t in tasks if t] + [g.strip() for g in gateways if g.strip()]
    
    return elements


def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text


def calculate_cosine_similarity(process_a_elements, process_b_elements):
    # Preprocessamento delle etichette
    process_a_elements = [preprocess_text(elem) for elem in process_a_elements]
    process_b_elements = [preprocess_text(elem) for elem in process_b_elements]

    all_elements = process_a_elements + process_b_elements

  
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_elements)
    
  
    process_a_matrix = tfidf_matrix[:len(process_a_elements)]
    process_b_matrix = tfidf_matrix[len(process_a_elements):]

    similarity_matrix = cosine_similarity(process_a_matrix, process_b_matrix)
    
    return similarity_matrix

def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività/gateway in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività/gateway in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score


def main(bpmn_file_a, bpmn_file_b):
    # Estrazione delle attività e dei gateway dai due file BPMN
    process_a_elements = extract_elements_from_bpmn(bpmn_file_a)
    process_b_elements = extract_elements_from_bpmn(bpmn_file_b)

    # Calcolo della cosine similarity
    similarity_matrix = calculate_cosine_similarity(process_a_elements, process_b_elements)
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
    # Visualizzazione del risultato
    print("Similarità tra le attività e i gateway:")
    for i, elem_a in enumerate(process_a_elements):
        for j, elem_b in enumerate(process_b_elements):
            print(f"'{elem_a}' vs '{elem_b}' = Similarity: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")


bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'
main(bpmn_file_a, bpmn_file_b)
# In[ ]:




