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

# Funzione principale
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

# Esempio di utilizzo con due file BPMN
bpmn_file_a = 'diagram (5).bpmn'
bpmn_file_b = 'diagram (6).bpmn'

main(bpmn_file_a, bpmn_file_b)


# In[6]:


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


# In[1]:


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

def test_glove_model(embedding_model):
    test_words = ['task', 'process', 'action']  # Parole di esempio
    for word in test_words:
        if word in embedding_model:
            print(f"'{word}' trovato nel modello GloVe")
        else:
            print(f"'{word}' NON trovato nel modello GloVe")

# Carica e verifica il modello GloVe
embedding_model = load_glove_model('glove.6B.300d.txt')
test_glove_model(embedding_model)


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


# In[14]:


import xml.etree.ElementTree as ET
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors

# Funzione per estrarre le attività da un file BPMN
def extract_tasks_from_bpmn(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei nomi delle attività dai nodi <bpmn:task>
    tasks = [elem.get('name') for elem in root.findall('.//bpmn:task', namespace)]
    return [t for t in tasks if t]  # Ritorna solo le attività non vuote

# Funzione per preprocessare il testo senza rimuovere i numeri
def preprocess_text(text):
    text = text.lower()  # Conversione in minuscolo
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Mantiene lettere e numeri, rimuove solo caratteri non alfanumerici
    return text

# Funzione per calcolare l'embedding medio di una frase
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

# Funzione per calcolare la cosine similarity tra due processi
def calculate_cosine_similarity(process_a_tasks, process_b_tasks, embedding_model):
    # Preprocessamento delle etichette
    process_a_tasks = [preprocess_text(task) for task in process_a_tasks]
    process_b_tasks = [preprocess_text(task) for task in process_b_tasks]

    # Calcoliamo gli embedding per ogni attività
    process_a_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_a_tasks]
    process_b_embeddings = [get_sentence_embedding(task, embedding_model) for task in process_b_tasks]

    # Calcolo della cosine similarity tra gli embedding
    similarity_matrix = cosine_similarity(process_a_embeddings, process_b_embeddings)
    
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

# Funzione principale per eseguire il confronto tra i file BPMN
def compare_bpmn_processes(bpmn_file_a, bpmn_file_b, embedding_model_file):
    # Carica il modello di word embedding (es. GloVe)
    print("Loading GloVe model...")
    embedding_model = KeyedVectors.load_word2vec_format(embedding_model_file, binary=False)
    print("GloVe model loaded with", len(embedding_model.index_to_key), "words.")
    
    # Estrazione delle attività dai due file BPMN
    process_a_tasks = extract_tasks_from_bpmn(bpmn_file_a)
    process_b_tasks = extract_tasks_from_bpmn(bpmn_file_b)
    
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

# Esempio di utilizzo con due file BPMN e il modello GloVe pre-addestrato
bpmn_file_a = 'diagram (5).bpmn'  # Cambia il nome del file con quello reale
bpmn_file_b = 'diagram (6).bpmn'  # Cambia il nome del file con quello reale
embedding_model_file = 'glove.6B.300d.txt'  # Modello GloVe pre-addestrato (testo)

# Esegui il confronto tra i processi BPMN
compare_bpmn_processes(bpmn_file_a, bpmn_file_b, embedding_model_file)


# In[ ]:




