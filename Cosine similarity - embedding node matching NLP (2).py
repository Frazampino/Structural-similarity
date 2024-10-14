#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
    print("Similarity betweeen activities:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")

# Esempio di utilizzo con due file BPMN
bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'

main(bpmn_file_a, bpmn_file_b)


# In[1]:


# Funzione per estrarre le attività e i gateway da un file BPMN
def extract_elements_from_bpmn(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei nomi delle attività dai nodi <bpmn:task>
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    
    # Estrazione dei gateway (exclusive, parallel, inclusive)
    gateways = []
    gateway_types = ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway']
    
    for g_type in gateway_types:
        gateways += [g_type + " " + (elem.get('name') if elem.get('name') else '') 
                     for elem in root.findall(f'.//bpmn:{g_type}', namespace)]
    
    # Filtra le attività e gateway non vuoti
    elements = [t for t in tasks if t] + [g.strip() for g in gateways if g.strip()]
    
    return elements


# In[3]:


import xml.etree.ElementTree as ET
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Funzione per estrarre le attività e i gateway da un file BPMN
def extract_elements_from_bpmn(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei nomi delle attività dai nodi <bpmn:task>
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    
    # Estrazione dei gateway (exclusive, parallel, inclusive)
    gateways = []
    gateway_types = ['exclusiveGateway', 'parallelGateway', 'inclusiveGateway']
    
    for g_type in gateway_types:
        gateways += [g_type + " " + (elem.get('name') if elem.get('name') else '') 
                     for elem in root.findall(f'.//bpmn:{g_type}', namespace)]
    
    # Filtra le attività e gateway non vuoti
    elements = [t for t in tasks if t] + [g.strip() for g in gateways if g.strip()]
    
    return elements

# Funzione per preprocessare il testo
def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text

# Funzione per calcolare la cosine similarity tra due processi
def calculate_cosine_similarity(process_a_elements, process_b_elements):
    # Preprocessamento delle etichette
    process_a_elements = [preprocess_text(elem) for elem in process_a_elements]
    process_b_elements = [preprocess_text(elem) for elem in process_b_elements]

    # Uniamo le attività in un'unica lista per la vectorizzazione
    all_elements = process_a_elements + process_b_elements

    # Calcolo della TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_elements)
    
    # Separiamo le matrici TF-IDF per ciascun processo
    process_a_matrix = tfidf_matrix[:len(process_a_elements)]
    process_b_matrix = tfidf_matrix[len(process_a_elements):]

    # Calcolo della cosine similarity tra le attività e i gateway dei due processi
    similarity_matrix = cosine_similarity(process_a_matrix, process_b_matrix)
    
    return similarity_matrix

# Funzione per calcolare il Node Matching Score
def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività/gateway in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività/gateway in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

# Funzione principale
def main(bpmn_file_a, bpmn_file_b):
    # Estrazione delle attività e dei gateway dai due file BPMN
    process_a_elements = extract_elements_from_bpmn(bpmn_file_a)
    process_b_elements = extract_elements_from_bpmn(bpmn_file_b)

    # Calcolo della cosine similarity
    similarity_matrix = calculate_cosine_similarity(process_a_elements, process_b_elements)
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
    # Visualizzazione del risultato
    print("Similarity between activities and gateway:")
    for i, elem_a in enumerate(process_a_elements):
        for j, elem_b in enumerate(process_b_elements):
            print(f"'{elem_a}' vs '{elem_b}' = Similarity: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")

# Esempio di utilizzo con due file BPMN
bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'
main(bpmn_file_a, bpmn_file_b)


# In[12]:


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
    bpmn_file_a = 'diagram (7) (1).bpmn'
    bpmn_file_b = 'diagram (7).bpmn'
    
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


# In[5]:


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
    bpmn_file_a = 'diagram (6).bpmn'
    bpmn_file_b = 'diagram (6) (1).bpmn'
    
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


# In[7]:


pip install gensim


# In[16]:


from gensim.scripts.glove2word2vec import glove2word2vec

# Specifica il file di input GloVe e il file di output
glove_input_file = 'glove.6B.300d.txt'
word2vec_output_file = 'glove.6B.300d.word2vec.txt'
glove2word2vec(glove_input_file, word2vec_output_file)


# In[17]:


from gensim.models import KeyedVectors

# Carica il modello convertito
embedding_model = KeyedVectors.load_word2vec_format('glove.6B.300d.word2vec.txt', binary=False)


# In[19]:


def load_glove_model(glove_file):
    """Carica il modello GloVe in un dizionario."""
    print("Loading GloVe model...")
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print("GloVe model loaded.")
    return glove_model


# In[20]:


def get_sentence_embedding(sentence, embedding_model):
    words = sentence.split()  # Dividiamo la frase in parole
    word_embeddings = []
    
    # Per ogni parola, se presente nel modello, otteniamo il suo embedding
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
    
    if word_embeddings:
        # Restituiamo la media degli embedding delle parole
        return np.mean(word_embeddings, axis=0)
    else:
        # Se nessuna parola ha un embedding, restituiamo un vettore nullo
        return np.zeros(300)  # GloVe 300-dimensional


# In[21]:


def main(bpmn_file_a, bpmn_file_b, glove_file):
    # Carica il modello GloVe
    embedding_model = load_glove_model(glove_file)
    
    # Estrazione delle attività dai due file BPMN
    process_a_tasks = extract_tasks_from_bpmn(bpmn_file_a)
    process_b_tasks = extract_tasks_from_bpmn(bpmn_file_b)

    # Calcolo della cosine similarity usando gli embedding
    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model)
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
    # Visualizzazione del risultato
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")


# In[22]:


bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'

glove_file = 'glove.6B.300d.txt'  # File GloVe pre-addestrato

main(bpmn_file_a, bpmn_file_b, glove_file)


# In[23]:


def test_glove_loading(embedding_model):
    test_words = ['task', 'process', 'action']  # Alcune parole di test
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            print(f"'{word}' NON trovato nel modello GloVe")

# Chiamalo dopo aver caricato il modello
test_glove_loading(embedding_model)


# In[29]:


def check_embeddings(embedding_model, sentences):
    for sentence in sentences:
        embedding = get_sentence_embedding(sentence, embedding_model)
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase: '{sentence}'")


# In[30]:


def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score


# In[31]:


def calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model):
    process_a_tasks = [preprocess_text(task) for task in process_a_tasks]
    process_b_tasks = [preprocess_text(task) for task in process_b_tasks]

    process_a_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_a_tasks]
    process_b_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_b_tasks]

    # Verifica che le embeddings non siano tutte nulle
    for i, embedding in enumerate(process_a_embeddings):
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase in Process A: '{process_a_tasks[i]}'")
    for i, embedding in enumerate(process_b_embeddings):
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase in Process B: '{process_b_tasks[i]}'")
    
    similarity_matrix = cosine_similarity(process_a_embeddings, process_b_embeddings)
    
    return similarity_matrix


# In[32]:


def calculate_node_matching_score(similarity_matrix):
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Verifica i punteggi di similarità massimi
    print("Similarità massime da Process A a Process B:", max_similarities_a_to_b)
    print("Similarità massime da Process B a Process A:", max_similarities_b_to_a)
    
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score


# In[33]:


def main(bpmn_file_a, bpmn_file_b, glove_file):
    embedding_model = load_glove_model(glove_file)
    
    process_a_tasks = extract_tasks_from_bpmn(bpmn_file_a)
    process_b_tasks = extract_tasks_from_bpmn(bpmn_file_b)

    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model)
    
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")


# In[35]:


bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'
glove_file = 'glove.6B.300d.txt'

main(bpmn_file_a, bpmn_file_b, glove_file)


# In[36]:


def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text


# In[37]:


def test_glove_loading(embedding_model):
    test_words = ['task', 'process', 'action']  # Alcune parole di test
    not_found = []
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            not_found.append(word)
            print(f"'{word}' NON trovato nel modello GloVe")
    if not_found:
        print("Parole non trovate nel modello GloVe:", not_found)


# In[38]:


def test_sentences():
    test_sentences = ["task1", "task2", "task3"]
    for sentence in test_sentences:
        embedding = get_sentence_embedding(sentence, embedding_model)
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase: '{sentence}'")
        else:
            print(f"Embedding per '{sentence}': {embedding[:5]}...")  # Stampa solo i primi 5 valori per brevità

# Carica il modello e testa le frasi
embedding_model = load_glove_model('glove.6B.300d.txt')
test_sentences()


# In[39]:


def test_glove_loading(embedding_model):
    test_words = ['the', 'and', 'of']  # Alcune parole molto comuni
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            print(f"'{word}' NON trovato nel modello GloVe")

# Carica il modello e testalo
embedding_model = load_glove_model('glove.6B.300d.txt')
test_glove_loading(embedding_model)


# In[2]:


import numpy as np
import re

def load_glove_model(glove_file):
    print("Loading GloVe model...")
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print(f"GloVe model loaded with {len(glove_model)} words.")
    return glove_model

def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text

def get_sentence_embedding(sentence, embedding_model):
    words = preprocess_text(sentence).split()
    word_embeddings = []
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
        else:
            print(f"'{word}' non trovato nel modello GloVe")
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)  # GloVe 300-dimensional

def test_glove_model(embedding_model):
    test_words = ['task', 'process', 'action']  # Parole di esempio
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            print(f"'{word}' NON trovato nel modello GloVe")

def test_preprocessing():
    test_sentences = ["Task1", "task1", "TASK1", "task2", "task3"]
    for sentence in test_sentences:
        processed = preprocess_text(sentence)
        print(f"Original: '{sentence}' -> Preprocessed: '{processed}'")

def test_sentence_embeddings():
    test_sentences = ["task1", "task2", "task3"]
    for sentence in test_sentences:
        embedding = get_sentence_embedding(sentence, embedding_model)
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase: '{sentence}'")
        else:
            print(f"Embedding per '{sentence}': {embedding[:5]}...")  # Stampa solo i primi 5 valori per brevità

# Carica e verifica il modello GloVe
embedding_model = load_glove_model('glove.6B.300d.txt')
test_glove_model(embedding_model)
test_preprocessing()
test_sentence_embeddings()


# In[3]:


def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\d+', '', text)  # Rimozione dei numeri
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text.strip()  # Rimozione degli spazi finali


# In[4]:


def get_sentence_embedding(sentence, embedding_model):
    words = preprocess_text(sentence).split()
    word_embeddings = []
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
        else:
            print(f"'{word}' non trovato nel modello GloVe, uso 'task' se disponibile.")
            if 'task' in embedding_model:
                word_embeddings.append(embedding_model['task'])  # Usa "task" come fallback
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)  # GloVe 300-dimensional


# In[5]:


import numpy as np
import re

def load_glove_model(glove_file):
    print("Loading GloVe model...")
    glove_model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array([float(val) for val in split_line[1:]])
            glove_model[word] = embedding
    print(f"GloVe model loaded with {len(glove_model)} words.")
    return glove_model

def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'\d+', '', text)  # Rimozione dei numeri
    text = re.sub(r'\W+', ' ', text)  # Rimozione dei caratteri non alfanumerici
    return text.strip()  # Rimozione degli spazi finali

def get_sentence_embedding(sentence, embedding_model):
    words = preprocess_text(sentence).split()
    word_embeddings = []
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
        else:
            print(f"'{word}' non trovato nel modello GloVe, uso 'task' se disponibile.")
            if 'task' in embedding_model:
                word_embeddings.append(embedding_model['task'])  # Usa "task" come fallback
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(300)  # GloVe 300-dimensional

def test_glove_model(embedding_model):
    test_words = ['task', 'process', 'action']  # Parole di esempio
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            print(f"'{word}' NON trovato nel modello GloVe")

def test_preprocessing():
    test_sentences = ["Task1", "task1", "TASK1", "task2", "task3"]
    for sentence in test_sentences:
        processed = preprocess_text(sentence)
        print(f"Original: '{sentence}' -> Preprocessed: '{processed}'")

def test_sentence_embeddings():
    test_sentences = ["task1", "task2", "task3"]
    for sentence in test_sentences:
        embedding = get_sentence_embedding(sentence, embedding_model)
        if np.all(embedding == 0):
            print(f"Embedding nullo per la frase: '{sentence}'")
        else:
            print(f"Embedding per '{sentence}': {embedding[:5]}...")  # Stampa solo i primi 5 valori per brevità

# Carica e verifica il modello GloVe
embedding_model = load_glove_model('glove.6B.300d.txt')
test_glove_model(embedding_model)
test_preprocessing()
test_sentence_embeddings()


# In[6]:


from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model):
    # Preprocessamento delle etichette e calcolo degli embedding per ogni attività
    process_a_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_a_tasks]
    process_b_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_b_tasks]

    # Calcolo della cosine similarity tra gli embedding
    similarity_matrix = cosine_similarity(process_a_embeddings, process_b_embeddings)
    
    return similarity_matrix

def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

# Funzione principale per eseguire il confronto tra i processi
def compare_processes(process_a_tasks, process_b_tasks, embedding_model):
    # Calcolo della cosine similarity
    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model)
    
    # Stampa della matrice di similarità
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")
    
    return node_matching_score

# Definire le attività dei due processi (come esempio)
process_a_tasks = ["task1", "task2", "task3"]  # Attività di esempio del processo A
process_b_tasks = ["task1", "task2", "task3", "task4"]  # Attività di esempio del processo B

# Confronto dei due processi
compare_processes(process_a_tasks, process_b_tasks, embedding_model)


# In[7]:


# Funzione per preprocessare il testo senza rimuovere i numeri
def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Rimuove solo caratteri non alfanumerici, mantenendo i numeri
    return text


# In[9]:


def get_sentence_embedding(sentence, embedding_model):
    words = sentence.split()  # Dividiamo la frase in parole
    word_embeddings = []
    
    # Per ogni parola, se presente nel modello, otteniamo il suo embedding
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
        else:
            # Se la parola contiene numeri (es. 'task1'), separiamo la parte testuale dal numero
            parts = re.split('(\d+)', word)
            for part in parts:
                if part in embedding_model:
                    word_embeddings.append(embedding_model[part])
    
    if word_embeddings:
        # Restituiamo la media degli embedding delle parole
        return np.mean(word_embeddings, axis=0)
    else:
        # Se nessuna parola ha un embedding, restituiamo un vettore nullo
        return np.zeros(embedding_model.vector_size)


# In[10]:


from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model):
    # Preprocessamento delle etichette e calcolo degli embedding per ogni attività
    process_a_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_a_tasks]
    process_b_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_b_tasks]

    # Calcolo della cosine similarity tra gli embedding
    similarity_matrix = cosine_similarity(process_a_embeddings, process_b_embeddings)
    
    return similarity_matrix

def calculate_node_matching_score(similarity_matrix):
    # Per ogni attività in Process A, troviamo il massimo punteggio di similarità con Process B
    max_similarities_a_to_b = similarity_matrix.max(axis=1)
    
    # Per ogni attività in Process B, troviamo il massimo punteggio di similarità con Process A
    max_similarities_b_to_a = similarity_matrix.max(axis=0)
    
    # Calcolo della media delle similarità massime
    final_score = (np.mean(max_similarities_a_to_b) + np.mean(max_similarities_b_to_a)) / 2
    
    return final_score

# Funzione principale per eseguire il confronto tra i processi
def compare_processes(process_a_tasks, process_b_tasks, embedding_model):
    # Calcolo della cosine similarity
    similarity_matrix = calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model)
    
    # Stampa della matrice di similarità
    print("Similarità tra le attività:")
    for i, task_a in enumerate(process_a_tasks):
        for j, task_b in enumerate(process_b_tasks):
            print(f"'{task_a}' vs '{task_b}' = Similarità: {similarity_matrix[i, j]:.4f}")
    
    # Calcolo del Node Matching Score
    node_matching_score = calculate_node_matching_score(similarity_matrix)
    print(f"\nNode Matching Score Finale: {node_matching_score:.4f}")
    
    return node_matching_score


# In[11]:


# Definire le attività dei due processi (come esempio)
process_a_tasks = ["task1", "task2", "task3"]  # Attività di esempio del processo A
process_b_tasks = ["task1", "task2", "task3", "task4"]  # Attività di esempio del processo B

# Confronto dei due processi
compare_processes(process_a_tasks, process_b_tasks, embedding_model)


# In[17]:


# Definire le attività dei due processi (come esempio)
process_a_tasks = ["order", "process", "deliver"]  # Attività di esempio del processo A
process_b_tasks = ["ordering", "process", "production", "delivering"]  # Attività di esempio del processo B

# Confronto dei due processi
compare_processes(process_a_tasks, process_b_tasks, embedding_model)


# In[13]:


def get_sentence_embedding(sentence, embedding_model):
    words = sentence.split()  # Dividiamo la frase in parole
    word_embeddings = []
    
    # Per ogni parola, se presente nel modello, otteniamo il suo embedding
    for word in words:
        if word in embedding_model:
            word_embeddings.append(embedding_model[word])
        else:
            # Se la parola contiene numeri (es. 'task1'), separiamo la parte testuale dal numero
            parts = re.split('(\d+)', word)
            for part in parts:
                if part.isdigit():  # Controlliamo se 'part' è un numero
                    continue  # Ignoriamo la parte numerica
                elif part in embedding_model:
                    word_embeddings.append(embedding_model[part])
    
    if word_embeddings:
        # Restituiamo la media degli embedding delle parole
        return np.mean(word_embeddings, axis=0)
    else:
        # Se nessuna parola ha un embedding, restituiamo un vettore nullo
        return np.zeros(embedding_model.vector_size)


# In[16]:


import xml.etree.ElementTree as ET

def extract_labels_from_bpmn(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Namespace di BPMN
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione delle etichette (es. nome delle attività)
    labels = []
    for element in root.findall(".//bpmn:task", namespace):
        label = element.get('name')
        if label:
            labels.append(label)
    
    return labels

# Caricare etichette dai file BPMN
labels_file1 = extract_labels_from_bpmn('diagram (5).bpmn')
labels_file2 = extract_labels_from_bpmn('diagram (7) (1).bpmn')

print(labels_file1, labels_file2)


# In[17]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scaricare risorse necessarie
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('italian'))

def preprocess_labels(labels):
    processed_labels = []
    for label in labels:
        # Tokenizzazione
        tokens = word_tokenize(label.lower())
        # Rimozione stop words e lemmatizzazione
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
        processed_labels.append(" ".join(tokens))
    return processed_labels

# Pre-elaborare le etichette
labels_file1 = preprocess_labels(labels_file1)
labels_file2 = preprocess_labels(labels_file2)

print(labels_file1, labels_file2)


# In[18]:


from gensim.models import Word2Vec

# Tokenizzazione delle etichette per l'addestramento
tokenized_labels = [label.split() for label in (labels_file1 + labels_file2)]

# Addestramento modello Word2Vec
model = Word2Vec(sentences=tokenized_labels, vector_size=100, window=5, min_count=1, workers=4)

# Recuperare vettori per le parole chiave nelle etichette
def get_label_vector(label, model):
    words = label.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    
    # Media dei vettori per ottenere la rappresentazione dell'etichetta
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return None  # Gestione del caso in cui non ci siano parole nel vocabolario

label_vectors_file1 = [get_label_vector(label, model) for label in labels_file1]
label_vectors_file2 = [get_label_vector(label, model) for label in labels_file2]

print(label_vectors_file1, label_vectors_file2)


# In[19]:


# Addestramento Word2Vec su un corpus personalizzato
model = Word2Vec(sentences=tokenized_labels, vector_size=100, window=5, min_count=1, workers=4)


# In[20]:


from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(label_vectors1, label_vectors2):
    similarities = []
    for vec1 in label_vectors1:
        for vec2 in label_vectors2:
            if vec1 is not None and vec2 is not None:
                sim = cosine_similarity([vec1], [vec2])[0][0]
                similarities.append(sim)
    return similarities

similarities = calculate_similarity(label_vectors_file1, label_vectors_file2)
print(similarities)

#Valori alti (vicini a 1) indicano che le attività dei due processi sono molto simili.
#Valori bassi (vicini a 0) indicano che le attività sono poco simili o a diversa granularità.


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_similarity(label_vectors1, label_vectors2, aggregate='average'):
    similarities = []
    
    # Calculate pairwise cosine similarities
    for vec1 in label_vectors1:
        for vec2 in label_vectors2:
            if vec1 is not None and vec2 is not None:
                sim = cosine_similarity([vec1], [vec2])[0][0]
                similarities.append(sim)

    # Aggregate the similarities based on the specified method
    if similarities:  # Check if the list is not empty
        if aggregate == 'average':
            return np.mean(similarities)
        elif aggregate == 'max':
            return np.max(similarities)
        elif aggregate == 'total':
            return np.sum(similarities)
        else:
            raise ValueError("Invalid aggregate method. Choose 'average', 'max', or 'total'.")
    else:
        return 0  # Return 0 if there are no valid similarities

# Example usage
similarities = calculate_similarity(label_vectors_file1, label_vectors_file2, aggregate='average')
print(similarities)


# In[22]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

def calculate_similarity(label_vectors1, label_vectors2):
    similarities = []
    
    for vec1 in label_vectors1:
        for vec2 in label_vectors2:
            if vec1 is not None and vec2 is not None:
                sim = cosine_similarity([vec1], [vec2])[0][0]
                similarities.append(sim)

    return similarities


similarities = calculate_similarity(label_vectors_file1, label_vectors_file2)


mean_similarity = np.mean(similarities)
std_similarity = np.std(similarities)

print("Media delle similarità:", mean_similarity)
print("Deviazione Standard delle similarità:", std_similarity)


plt.hist(similarities, bins=20, edgecolor='k')
plt.title('Distribuzione delle Similarità Coseno')
plt.xlabel('Similarità Coseno')
plt.ylabel('Frequenza')
plt.show()


# In[3]:


get_ipython().system('pip install --upgrade pm4py')

# Importa i pacchetti necessari
from pm4py.objects.bpmn.importer import importer as bpmn_importer


# In[1]:


# Installazione di PM4Py
get_ipython().system('pip install --upgrade pm4py')

# Importa i pacchetti necessari
from pm4py.objects.bpmn.importer import importer as bpmn_importer
from pm4py.algo.discovery.footprints import factory as fp_discovery
from pm4py.objects.bpmn.visualization import factory as bpmn_visualization
from google.colab import files

# Carica due file BPMN
uploaded = files.upload()

# Ottieni i nomi dei file caricati
bpmn_file_names = list(uploaded.keys())

# Assicurati di avere esattamente due file
if len(bpmn_file_names) != 2:
    raise ValueError("Devi caricare esattamente due file BPMN.")

# Importa i modelli BPMN
bpmn_model_1 = bpmn_importer.apply(bpmn_file_names[0])
bpmn_model_2 = bpmn_importer.apply(bpmn_file_names[1])

# Calcola i Casual Footprints per entrambi i modelli
fp_model_1 = fp_discovery.apply(bpmn_model_1)
fp_model_2 = fp_discovery.apply(bpmn_model_2)

# Stampa i Casual Footprints
print("Casual Footprints del Modello 1:", fp_model_1)
print("Casual Footprints del Modello 2:", fp_model_2)

# Analisi della similarità basata su casual footprints
def compare_footprints(fp1, fp2):
    common = fp1.keys() & fp2.keys()  # Attività comuni
    similarities = {activity: (fp1[activity], fp2[activity]) for activity in common}
    return similarities

similarities = compare_footprints(fp_model_1, fp_model_2)
print("\nSimilarità tra i footprints dei modelli:")
for activity, footprint in similarities.items():
    print(f"Attività: {activity}, Footprint 1: {footprint[0]}, Footprint 2: {footprint[1]}")


# In[1]:


import xml.etree.ElementTree as ET

# Funzione per parsare il file BPMN
def parse_bpmn(file_path):
    # Legge il file BPMN come XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Definizione del namespace per trovare gli elementi BPMN
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    tasks = []
    events = []
    gateways = []

    # Parse dei task BPMN
    for task in root.findall('.//bpmn:task', ns):
        tasks.append(task.attrib['id'])

    # Parse degli eventi BPMN
    for event in root.findall('.//bpmn:startEvent', ns):
        events.append(('start', event.attrib['id']))

    for event in root.findall('.//bpmn:endEvent', ns):
        events.append(('end', event.attrib['id']))

    # Parse dei gateway BPMN
    for gateway in root.findall('.//bpmn:exclusiveGateway', ns):
        gateways.append(('xor', gateway.attrib['id']))
    
    for gateway in root.findall('.//bpmn:parallelGateway', ns):
        gateways.append(('and', gateway.attrib['id']))

    return tasks, events, gateways

# Classe per costruire la Rete di Petri
class PetriNet:
    def __init__(self):
        self.places = set()
        self.transitions = set()
        self.arcs = []

    def add_place(self, place):
        self.places.add(place)

    def add_transition(self, transition):
        self.transitions.add(transition)

    def add_arc(self, from_element, to_element):
        self.arcs.append((from_element, to_element))

    def display(self):
        print("Places:", self.places)
        print("Transitions:", self.transitions)
        print("Arcs:")
        for arc in self.arcs:
            print(f"{arc[0]} -> {arc[1]}")

# Funzione per convertire BPMN in Petri Net
def bpmn_to_petri_net(tasks, events, gateways):
    net = PetriNet()

    # Aggiungi posti per gli eventi (eventi di inizio e fine)
    for event in events:
        net.add_place(event[1])

    # Aggiungi transizioni per i task
    for task in tasks:
        net.add_transition(task)

    # Aggiungi gateway come transizioni
    for gateway in gateways:
        net.add_transition(gateway[1])

    # Mappa un esempio semplice di flusso
    # In questo esempio, mappiamo l'evento di inizio a una transizione (task), e poi all'evento di fine
    if events:
        start_event = events[0][1]  # Evento di inizio
        end_event = events[-1][1]  # Evento di fine

        if tasks:
            first_task = tasks[0]
            last_task = tasks[-1]

            # Collega l'evento di inizio al primo task
            net.add_arc(start_event, first_task)
            # Collega l'ultimo task all'evento di fine
            net.add_arc(last_task, end_event)

    return net

# Punto di ingresso: caricare il file BPMN e convertirlo
if __name__ == "__main__":
    # Specifica il percorso del file BPMN
    file_path = 'diagram (5).bpmn'  # Assicurati che il file sia in questa posizione o specifica il percorso corretto

    # Parse il file BPMN
    tasks, events, gateways = parse_bpmn(file_path)

    # Converti BPMN in una Rete di Petri
    petri_net = bpmn_to_petri_net(tasks, events, gateways)

    # Visualizza la Rete di Petri
    petri_net.display()


# In[2]:


import xml.etree.ElementTree as ET

# Funzione per parsare il file BPMN
def parse_bpmn(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Namespace del BPMN
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    tasks = []
    events = []
    gateways = []

    # Parse dei task BPMN
    for task in root.findall('.//bpmn:task', ns):
        tasks.append(task.attrib['id'])

    # Parse degli eventi BPMN (eventi di inizio e fine)
    for event in root.findall('.//bpmn:startEvent', ns):
        events.append(('start', event.attrib['id']))

    for event in root.findall('.//bpmn:endEvent', ns):
        events.append(('end', event.attrib['id']))

    # Parse dei gateway BPMN (XOR e AND)
    for gateway in root.findall('.//bpmn:exclusiveGateway', ns):
        gateways.append(('xor', gateway.attrib['id']))
    
    for gateway in root.findall('.//bpmn:parallelGateway', ns):
        gateways.append(('and', gateway.attrib['id']))

    return tasks, events, gateways

# Classe per gestire la Rete di Petri
class PetriNet:
    def __init__(self):
        self.places = set()
        self.transitions = set()
        self.arcs = []

    def add_place(self, place):
        self.places.add(place)

    def add_transition(self, transition):
        self.transitions.add(transition)

    def add_arc(self, from_element, to_element):
        self.arcs.append((from_element, to_element))

    def display(self):
        print("Places:", self.places)
        print("Transitions:", self.transitions)
        print("Arcs:")
        for arc in self.arcs:
            print(f"{arc[0]} -> {arc[1]}")

# Funzione per convertire BPMN in una Rete di Petri
def bpmn_to_petri_net(tasks, events, gateways):
    net = PetriNet()

    # Aggiungi posti per gli eventi (eventi di inizio e fine)
    for event in events:
        net.add_place(event[1])

    # Aggiungi transizioni per i task
    for task in tasks:
        net.add_transition(task)

    # Aggiungi gateway come transizioni
    for gateway in gateways:
        net.add_transition(gateway[1])

    # Mappiamo un esempio semplice di flusso tra eventi di inizio e fine
    if events:
        start_event = events[0][1]  # Evento di inizio
        end_event = events[-1][1]  # Evento di fine

        if tasks:
            first_task = tasks[0]
            last_task = tasks[-1]

            # Collega l'evento di inizio al primo task
            net.add_arc(start_event, first_task)
            # Collega l'ultimo task all'evento di fine
            net.add_arc(last_task, end_event)

    return net

# Funzione per esportare la Rete di Petri in formato PNML
class PNMLExporter:
    def __init__(self, petri_net):
        self.petri_net = petri_net

    def export(self, file_path):
        pnml = ET.Element("pnml")
        net = ET.SubElement(pnml, "net", id="net1", type="http://www.pnml.org/version-2009/grammar/pnml")

        # Aggiungi posti
        for place in self.petri_net.places:
            place_elem = ET.SubElement(net, "place", id=place)
            name_elem = ET.SubElement(place_elem, "name")
            ET.SubElement(name_elem, "text").text = place

        # Aggiungi transizioni
        for transition in self.petri_net.transitions:
            trans_elem = ET.SubElement(net, "transition", id=transition)
            name_elem = ET.SubElement(trans_elem, "name")
            ET.SubElement(name_elem, "text").text = transition

        # Aggiungi archi
        for i, (source, target) in enumerate(self.petri_net.arcs):
            arc_elem = ET.SubElement(net, "arc", id=f"a{i+1}", source=source, target=target)

        # Scrivi l'albero XML nel file PNML
        tree = ET.ElementTree(pnml)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

        print(f"PNML file exported to: {file_path}")

# Punto di ingresso principale
if __name__ == "__main__":
    # Specifica il percorso del file BPMN
    bpmn_file_path = "diagram (5).bpmn"  # Inserisci il percorso del tuo file BPMN

    # Parse il file BPMN
    tasks, events, gateways = parse_bpmn(bpmn_file_path)

    # Converti BPMN in una Rete di Petri
    petri_net = bpmn_to_petri_net(tasks, events, gateways)

    # Visualizza la Rete di Petri
    petri_net.display()

    # Esporta la Rete di Petri in un file PNML
    pnml_file_path = "petri_net_output.pnml"  # Inserisci il nome del file PNML di output
    pnml_exporter = PNMLExporter(petri_net)
    pnml_exporter.export(pnml_file_path)


# In[ ]:




