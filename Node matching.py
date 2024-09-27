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
    
    # Estrazione dei nomi delle attività dai nodi <bpmn:task>
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    return [t for t in tasks if t]  # Ritorna solo le attività non vuote

# Funzione per preprocessare il testo
def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text

# Funzione per calcolare la cosine similarity tra due processi
def calculate_cosine_similarity(process_a_tasks, process_b_tasks):
    # Preprocessamento delle etichette
    process_a_tasks = [preprocess_text(task) for task in process_a_tasks]
    process_b_tasks = [preprocess_text(task) for task in process_b_tasks]

    # Uniamo le attività in un'unica lista per la vectorizzazione
    all_tasks = process_a_tasks + process_b_tasks

    # Calcolo della TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tasks)
    
    # Separiamo le matrici TF-IDF per ciascun processo
    process_a_matrix = tfidf_matrix[:len(process_a_tasks)]
    process_b_matrix = tfidf_matrix[len(process_a_tasks):]

    # Calcolo della cosine similarity tra le attività dei due processi
    similarity_matrix = cosine_similarity(process_a_matrix, process_b_matrix)
    
    return similarity_matrix

# Funzione per calcolare il Node Matching Score
def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

def main(bpmn_file_a, bpmn_file_b):
    # Estrazione delle attività dai due file BPMN
    process_a_tasks = extract_tasks_from_bpmn(bpmn_file_a)
    process_b_tasks = extract_tasks_from_bpmn(bpmn_file_b)

    # Calcolo della cosine similarity
    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks)
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
    # Visualizzazione del risultato
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")

# example
bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'

main(bpmn_file_a, bpmn_file_b)


# In[6]:
# In[ ]:




