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

import xml.etree.ElementTree as ET
import numpy as np

# Predefined similarity matrix between activities
similarity_matrix = {
    'Task1': {'Task1': 1.0, 'Task2': 0.0, 'Task3': 0.0, 'Task4': 0.0, 'exclusiveGateway': 0.0},
    'Task2': {'Task1': 0.0, 'Task2': 1.0, 'Task3': 0.0, 'Task4': 0.0, 'exclusiveGateway': 0.0},
    'Task3': {'Task1': 0.0, 'Task2': 0.0, 'Task3': 1.0, 'Task4': 0.0, 'exclusiveGateway': 0.0},
    'Task4': {'Task1': 0.0, 'Task2': 0.0, 'Task3': 0.0, 'Task4': 1.0, 'exclusiveGateway': 0.0},
    'exclusiveGateway': {'Task1': 0.0, 'Task2': 0.0, 'Task3': 0.0, 'Task4': 0.0, 'exclusiveGateway': 1.0}
}


def parse_bpmn(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
  
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
   
    tasks_sequence = []
    
    for task in root.findall('.//bpmn:task', ns):
        tasks_sequence.append(task.attrib['name'])  # Add task name
    
    for gateway in root.findall('.//bpmn:exclusiveGateway', ns):
        tasks_sequence.append('exclusiveGateway')  # Add gateway as a special entity
    
    return tasks_sequence


def create_similarity_matrix(sequence_a, sequence_b):
    matrix = np.zeros((len(sequence_a), len(sequence_b)))  # Initialize a matrix of zeros
    for i, task_a in enumerate(sequence_a):
        for j, task_b in enumerate(sequence_b):
            # Fetch similarity score from the predefined matrix
            matrix[i, j] = similarity_matrix.get(task_a, {}).get(task_b, 0.0)
    return matrix


def print_similarity_matrix(sequence_a, sequence_b, matrix):
    print("Similarity Matrix between BPMN A and BPMN B:")
    print(f"{'':<20}", end="")
    
    
    for task_b in sequence_b:
        print(f"{task_b:<20}", end="")
    print()
    
   
    for i, task_a in enumerate(sequence_a):
        print(f"{task_a:<20}", end="")
        for j in range(len(sequence_b)):
            print(f"{matrix[i, j]:<20.4f}", end="")
        print()


def calculate_node_matching_score(matrix):
    
    max_similarities_a_to_b = matrix.max(axis=1)
    
   
    max_similarities_b_to_a = matrix.max(axis=0)
    
   
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

# Main function to compare two BPMN files
def main():
    # File paths of the two BPMN diagrams
    bpmn_file_a = 'diagram (5).bpmn'
    bpmn_file_b = 'diagram (6).bpmn'
    
    try:
        
        sequence_a = parse_bpmn(bpmn_file_a)
        sequence_b = parse_bpmn(bpmn_file_b)
        
        print(f"Sequence extracted from BPMN A: {sequence_a}")
        print(f"Sequence extracted from BPMN B: {sequence_b}")
        
        
        similarity_matrix_result = create_similarity_matrix(sequence_a, sequence_b)
       
        print_similarity_matrix(sequence_a, sequence_b, similarity_matrix_result)
        
        
        node_matching_score = calculate_node_matching_score(similarity_matrix_result)
        print(f"\nNode Matching Score (similar to Prime Event Structure): {node_matching_score:.4f}")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")

# Run the main program
if __name__ == "__main__":
    main()


