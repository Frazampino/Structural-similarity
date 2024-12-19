#!/usr/bin/env python
# coding: utf-8

# In[5]:


#method Process Model Matching Contest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import xml.etree.ElementTree as ET

nltk.download('punkt')
nltk.download('wordnet')

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return ' '.join(lemmatized_tokens)

def extract_activities_from_bpmn(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Namespace BPMN
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    activities = []
    for task in root.findall('.//bpmn:task', namespace):
        activities.append(task.get('name', ''))
    
    return ' '.join(activities)

bpmn_file_1 = 'Signavio_source_partA.bpmn'
bpmn_file_2 = 'Signavio_source_partA (2).bpmn'

model_1 = extract_activities_from_bpmn(bpmn_file_1)
model_2 = extract_activities_from_bpmn(bpmn_file_2)

models = [lemmatize_text(model_1), lemmatize_text(model_2)]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(models)

cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print(f"Cosine similarity between processes: {cosine_sim[0][0]}")

threshold = 0.5
if cosine_sim[0][0] >= threshold:
    print("Models are similar")
else:
    print("Models are not similar.")


# In[10]:


## modello Sentence-BERT #stesso risultato di cosine similarity!! ma riduce di molto il lavoro
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-mpnet-base-v2')

bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA (2).bpmn"""

prompt = f"""Compare the following two BPMN models in XML format, paying special attention to the naming of tasks and how different names might represent similar processes. Assess their semantic similarity.

Modello 1:
{bpmn1}

Modello 2:
{bpmn2}

Please respond with a score from 0 to 1, where 0 means no similarity and 1 means perfect similarity, along with a brief explanation of the score. """

embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

print(f"Semantic similarity: {similarity.item()}")

#modello di linguaggio che utilizza una architettura di rete neurale Transformer per apprendere rappresentazioni semantiche del#
#linguaggio e può essere utilizzato per una varietà di compiti di linguaggio naturale.

#In particolare, MPNet è un esempio di LLM che utilizza una tecnica di pre-training chiamata "masked and permuted pre-training" per apprendere rappresentazioni semantiche del linguaggio. 


# In[3]:


## modello Sentence-BERT
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('nli-roberta-large')

bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA (2).bpmn"""

prompt = f"""Compare the following two BPMN models in XML format, paying special attention to the naming of tasks and how different names might represent similar processes. Assess their semantic similarity.

Modello 1:
{bpmn1}

Modello 2:
{bpmn2}

Please respond with a score from 0 to 1, where 0 means no similarity and 1 means perfect similarity, along with a brief explanation of the score."""

embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

print(f"Semantic similarity: {similarity.item()}")


# In[1]:


## modello Sentence-BERT
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-distilroberta-v1')

bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA (2).bpmn"""

prompt = f"""Compare the following two BPMN models in XML format, paying special attention to the naming of tasks and how different names might represent similar processes. Assess their semantic similarity.

Modello 1:
{bpmn1}

Modello 2:
{bpmn2}

Please respond with a score from 0 to 1, where 0 means no similarity and 1 means perfect similarity, along with a brief explanation of the score."""

embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

print(f"Semantic similarity: {similarity.item()}")


# In[8]:


from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('all-distilroberta-v1')

bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA false (1).bpmn"""

prompt = f"""Compare the following BPMN (Business Process Model and Notation) diagrams to measure their overall similarity. Provide a similarity value ranging from 0 to 1, where 0 indicates that the diagrams are completely different and 1 indicates that they are identical.

Include a detailed diagnostic of the differences by analyzing the following aspects:

Tasks: Compare the tasks present in both diagrams. Which tasks are common, and which are unique to each diagram? Also, analyze their complexity and role in the process.
Lanes: Examine the lanes used in the diagrams. Are the same lanes present in both diagrams? If there are differences, describe how these affect the structure and responsibility of the process.
Gateways: Analyze the gateways present in each diagram. What types of gateways are used (e.g., exclusive, parallel, inclusive)? Are there significant differences in how the gateways influence the flow of the process?
Provide an overall analysis that highlights the similarities and differences between the two BPMN diagrams, supported by specific examples.

Modello 1:
{bpmn1}

Modello 2:
{bpmn2}
Provide an overall analysis that highlights the similarities and differences between the two BPMN diagrams, supported by specific examples.

Please respond with a score from 0 to 1, where 0 means no similarity and 1 means perfect similarity, along with a brief explanation of the score."""

embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
 
print(f"similarity: {similarity.item()}")
response_content = {
    'message': {
        'content': f"similarity: {similarity.item()}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Overall Analysis: The diagrams have a similarity score of {similarity.item()}. "
                   f"While they share some common tasks and lanes, there are notable differences in the gateways used."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[10]:


from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-distilroberta-v1')

# Define the BPMN diagrams as strings
bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA false (1).bpmn"""

# Encode the BPMN diagrams to get their embeddings
embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

# Calculate the cosine similarity between the two embeddings
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Prepare the detailed diagnostic analysis
def analyze_bpmn(bpmn1, bpmn2):
    # Placeholder for analysis logic
    # You would need to implement actual parsing and comparison logic here
    analysis = {
        "tasks": {
            "common": ["Task A", "Task B"],  # Example common tasks
            "unique_to_bpmn1": ["Task C"],
            "unique_to_bpmn2": ["Task D"],
            "complexity": "Both diagrams have tasks of varying complexity, with Task A being the most complex."
        },
        "lanes": {
            "same_lanes": True,
            "differences": "Both diagrams use the same name lanes, which helps maintain clarity in responsibilities."
        },
        "gateways": {
            "types": ["exclusive", "parallel"],
            "differences": "BPMN1 uses an exclusive gateway for decision-making, while BPMN2 uses a parallel gateway."
        }
    }
    return analysis

# Perform the analysis
analysis_results = analyze_bpmn(bpmn1, bpmn2)

# Prepare the response content
response_content = {
    'message': {
        'content': f"Semantic similarity: {similarity.item()}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Overall Analysis: The diagrams have a similarity score of {similarity.item()}. "
                   f"While they share some common tasks and lanes, there are notable differences in the gateways used."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[11]:


from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-distilroberta-v1')

# Define the BPMN diagrams as strings
bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA false (1).bpmn"""

# Encode the BPMN diagrams to get their embeddings
embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

# Calculate the cosine similarity between the two embeddings
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Function to analyze BPMN diagrams
def analyze_bpmn(bpmn1, bpmn2):
    # Placeholder for actual parsing logic
    # Example lane names for demonstration
    lanes_bpmn1 = ["Lane A", "Lane B"]
    lanes_bpmn2 = ["Lane A", "Lane C"]  # Different lane name for demonstration

    # Tasks analysis (placeholder)
    tasks_analysis = {
        "common": ["Task A", "Task B"],
        "unique_to_bpmn1": ["Task C"],
        "unique_to_bpmn2": ["Task D"],
        "complexity": "Both diagrams have tasks of varying complexity, with Task A being the most complex."
    }

    # Lanes analysis
    same_lanes = set(lanes_bpmn1) == set(lanes_bpmn2)
    lane_differences = set(lanes_bpmn1).symmetric_difference(set(lanes_bpmn2))

    lanes_analysis = {
        "same_lanes": same_lanes,
        "differences": f"The following lanes are different: {', '.join(lane_differences)}" if lane_differences else "Both diagrams use the same lanes."
    }

    # Gateways analysis (placeholder)
    gateways_analysis = {
        "types": ["exclusive", "parallel"],
        "differences": "BPMN1 uses an exclusive gateway for decision-making, while BPMN2 uses a parallel gateway."
    }

    return {
        "tasks": tasks_analysis,
        "lanes": lanes_analysis,
        "gateways": gateways_analysis
    }

# Perform the analysis
analysis_results = analyze_bpmn(bpmn1, bpmn2)

# Prepare the response content
response_content = {
    'message': {
        'content': f"Semantic similarity: {similarity.item()}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Overall Analysis: The diagrams have a similarity score of {similarity.item()}. "
                   f"While they share some common tasks, there are notable differences in the lanes and gateways used."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[13]:


from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-distilroberta-v1')

# Define the BPMN diagrams as strings
bpmn1 = """Signavio_source_partA.bpmn"""
bpmn2 = """Signavio_source_partA false (1).bpmn"""

# Encode the BPMN diagrams to get their embeddings
embeddings = model.encode([bpmn1, bpmn2], convert_to_tensor=True)

# Calculate the cosine similarity between the two embeddings
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Function to analyze BPMN diagrams
def analyze_bpmn(bpmn1, bpmn2):
    # Placeholder for actual parsing logic
    # Example tasks for demonstration
    tasks_bpmn1 = ["Task A", "Task B", "Task C"]
    tasks_bpmn2 = ["Task A", "Task B", "Task D"]

    # Example lane names for demonstration
    lanes_bpmn1 = ["Lane A", "Lane B"]
    lanes_bpmn2 = ["Lane A", "Lane C"]  # Different lane name for demonstration

    # Example gateways for demonstration
    gateways_bpmn1 = ["Exclusive Gateway"]
    gateways_bpmn2 = ["Parallel Gateway"]

    # Tasks analysis
    common_tasks = set(tasks_bpmn1) & set(tasks_bpmn2)
    unique_to_bpmn1 = set(tasks_bpmn1) - set(tasks_bpmn2)
    unique_to_bpmn2 = set(tasks_bpmn2) - set(tasks_bpmn1)

    tasks_analysis = {
        "common": list(common_tasks),
        "unique_to_bpmn1": list(unique_to_bpmn1),
        "unique_to_bpmn2": list(unique_to_bpmn2),
        "complexity": "Both diagrams have tasks of varying complexity."
    }

    # Lanes analysis
    same_lanes = set(lanes_bpmn1) == set(lanes_bpmn2)
    lane_differences = set(lanes_bpmn1).symmetric_difference(set(lanes_bpmn2))

    lanes_analysis = {
        "same_lanes": same_lanes,
        "differences": f"The following lanes are different: {', '.join(lane_differences)}" if lane_differences else "Both diagrams use the same lanes."
    }

    # Gateways analysis
    gateway_differences = set(gateways_bpmn1).symmetric_difference(set(gateways_bpmn2))
    gateways_analysis = {
        "types": list(set(gateways_bpmn1) | set(gateways_bpmn2)),
        "differences": f"The following gateways are different: {', '.join(gateway_differences)}" if gateway_differences else "Both diagrams use the same gateways."
    }

    return {
        "tasks": tasks_analysis,
        "lanes": lanes_analysis,
        "gateways": gateways_analysis
    }

# Perform the analysis
analysis_results = analyze_bpmn(bpmn1, bpmn2)

# Prepare the response content
response_content = {
    'message': {
        'content': f"Similarity: {similarity.item()}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Overall Analysis: The diagrams have a similarity score of {similarity.item()}. "
                   f"While they share some common tasks, there are notable differences in the lanes and gateways used."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[15]:


import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-distilroberta-v1')

# Function to parse BPMN XML and extract tasks, lanes, and gateways
def parse_bpmn(file_path):
    tasks = []
    lanes = []
    gateways = []

    # Parse the BPMN XML content
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces if needed (adjust based on your BPMN XML)
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    # Extract tasks
    for task in root.findall('.//bpmn:task', namespaces):
        tasks.append(task.get('id'))  # or task.get('name') if available

    # Extract lanes
    for lane in root.findall('.//bpmn:lane', namespaces):
        lanes.append(lane.get('id'))  # or lane.get('name') if available

    # Extract gateways
    for gateway in root.findall('.//bpmn:gateway', namespaces):
        gateways.append(gateway.get('id'))  # or gateway.get('name') if available

    return tasks, lanes, gateways

# Load your BPMN files
bpmn1_file_path = "Signavio_source_partA.bpmn"  # Replace with the actual file path
bpmn2_file_path = "Signavio_source_partA false (1).bpmn"  # Replace with the actual file path

# Parse the BPMN diagrams
tasks_bpmn1, lanes_bpmn1, gateways_bpmn1 = parse_bpmn(bpmn1_file_path)
tasks_bpmn2, lanes_bpmn2, gateways_bpmn2 = parse_bpmn(bpmn2_file_path)

# Encode the BPMN diagrams to get their embeddings
with open(bpmn1_file_path, 'r') as file:
    bpmn1_content = file.read()
with open(bpmn2_file_path, 'r') as file:
    bpmn2_content = file.read()

embeddings = model.encode([bpmn1_content, bpmn2_content], convert_to_tensor=True)

# Calculate the cosine similarity between the two embeddings
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Function to analyze BPMN diagrams
def analyze_bpmn(tasks1, tasks2, lanes1, lanes2, gateways1, gateways2):
    # Tasks analysis
    common_tasks = set(tasks1) & set(tasks2)
    unique_to_bpmn1 = set(tasks1) - set(tasks2)
    unique_to_bpmn2 = set(tasks2) - set(tasks1)

    tasks_analysis = {
        "common": list(common_tasks),
        "unique_to_bpmn1": list(unique_to_bpmn1),
        "unique_to_bpmn2": list(unique_to_bpmn2),
        "complexity": "Both diagrams have tasks of varying complexity."
    }

    # Lanes analysis
    same_lanes = set(lanes1) == set(lanes2)
    lane_differences = set(lanes1).symmetric_difference(set(lanes2))

    lanes_analysis = {
        "same_lanes": same_lanes,
        "differences": f"The following lanes are different: {', '.join(lane_differences)}" if lane_differences else "Both diagrams use the same lanes."
    }

    # Gateways analysis
    gateway_differences = set(gateways1).symmetric_difference(set(gateways2))
    gateways_analysis = {
        "types": list(set(gateways1) | set(gateways2)),
        "differences": f"The following gateways are different: {', '.join(gateway_differences)}" if gateway_differences else "Both diagrams use the same gateways."
    }

    return {
        "tasks": tasks_analysis,
        "lanes": lanes_analysis,
        "gateways": gateways_analysis
    }

# Perform the analysis
analysis_results = analyze_bpmn(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2, gateways_bpmn1, gateways_bpmn2)

# Prepare the response content
response_content = {
    'message': {
        'content': f"Semantic similarity: {similarity.item()}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Overall Analysis: The diagrams have a similarity score of {similarity.item()}. "
                   f"While they share some common tasks, there are notable differences in the lanes and gateways used."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[25]:


import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util

# Load the SentenceTransformer model
model = SentenceTransformer('all-distilroberta-v1')

# Function to parse BPMN XML and extract tasks, lanes, and gateways
def parse_bpmn(file_path):
    tasks = []
    lanes = []
    gateways = []

    # Parse the BPMN XML content
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Define namespaces if needed (adjust based on your BPMN XML)
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    # Extract tasks (id and name if available)
    for task in root.findall('.//bpmn:task', namespaces):
        tasks.append({
            "id": task.get('id'),
            "name": task.get('name')
        })

    # Extract lanes (id and name if available)
    for lane in root.findall('.//bpmn:lane', namespaces):
        lanes.append({
            "id": lane.get('id'),
            "name": lane.get('name')
        })

    # Extract gateways (id and name if available)
    for gateway in root.findall('.//bpmn:gateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name')
        })

    return tasks, lanes, gateways

# Load your BPMN files
bpmn1_file_path = "Signavio_source_partA.bpmn"  # Replace with the actual file path
bpmn2_file_path = "Signavio_source_partA false (1).bpmn"  # Replace with the actual file path

# Parse the BPMN diagrams
tasks_bpmn1, lanes_bpmn1, gateways_bpmn1 = parse_bpmn(bpmn1_file_path)
tasks_bpmn2, lanes_bpmn2, gateways_bpmn2 = parse_bpmn(bpmn2_file_path)

# Encode the BPMN diagrams to get their embeddings
with open(bpmn1_file_path, 'r') as file:
    bpmn1_content = file.read()
with open(bpmn2_file_path, 'r') as file:
    bpmn2_content = file.read()

embeddings = model.encode([bpmn1_content, bpmn2_content], convert_to_tensor=True)

# Calculate the cosine similarity between the two embeddings
semantic_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Function to calculate semantic similarity between two strings
def calculate_similarity(str1, str2):
    return util.pytorch_cos_sim(model.encode(str1, convert_to_tensor=True), model.encode(str2, convert_to_tensor=True))

# Function to analyze BPMN diagrams for structural, semantic, and behavioral similarity
def analyze_bpmn(tasks1, tasks2, lanes1, lanes2, gateways1, gateways2):
    # Structural similarity
    structural_similarity = {
        "task_count": len(tasks1) == len(tasks2),
        "lane_count": len(lanes1) == len(lanes2),
        "gateway_count": len(gateways1) == len(gateways2)
    }

    # Tasks analysis
    tasks_bpmn1_ids = {task["id"] for task in tasks1}
    tasks_bpmn2_ids = {task["id"] for task in tasks2}
    tasks_bpmn1_names = {task["name"] for task in tasks1 if task["name"]}
    tasks_bpmn2_names = {task["name"] for task in tasks2 if task["name"]}

    common_tasks = tasks_bpmn1_ids & tasks_bpmn2_ids
    unique_to_bpmn1 = tasks_bpmn1_ids - tasks_bpmn2_ids
    unique_to_bpmn2 = tasks_bpmn2_ids - tasks_bpmn1_ids

    # Advanced task comparison: find similar but different tasks (using semantic similarity)
    similar_tasks = []
    for task1 in tasks1:
        for task2 in tasks2:
            if task1["id"] != task2["id"]:  # Ignore exact matches
                similarity_score = calculate_similarity(task1["name"], task2["name"])
                if similarity_score > 0.8:  # Threshold for "similar" tasks
                    similar_tasks.append({
                        "task1_id": task1["id"],
                        "task1_name": task1["name"],
                        "task2_id": task2["id"],
                        "task2_name": task2["name"],
                        "similarity": similarity_score.item()
                    })

    tasks_analysis = {
        "common_ids": list(common_tasks),
        "unique_to_bpmn1_ids": list(unique_to_bpmn1),
        "unique_to_bpmn2_ids": list(unique_to_bpmn2),
        "similar_tasks": similar_tasks,  # Include similar tasks
        "complexity": "Both diagrams have tasks of varying complexity."
    }

    # Lanes analysis (with semantic similarity for non-identical but similar lanes)
    lanes_bpmn1_ids = {lane["id"] for lane in lanes1}
    lanes_bpmn2_ids = {lane["id"] for lane in lanes2}
    lanes_bpmn1_names = {lane["name"] for lane in lanes1 if lane["name"]}
    lanes_bpmn2_names = {lane["name"] for lane in lanes2 if lane["name"]}

    common_lanes = lanes_bpmn1_ids & lanes_bpmn2_ids
    unique_to_bpmn1_lanes = lanes_bpmn1_ids - lanes_bpmn2_ids
    unique_to_bpmn2_lanes = lanes_bpmn2_ids - lanes_bpmn1_ids

    # Advanced lane comparison: find similar but different lanes (using semantic similarity)
    similar_lanes = []
    for lane1 in lanes1:
        for lane2 in lanes2:
            if lane1["id"] != lane2["id"]:  # Ignore exact matches
                similarity_score = calculate_similarity(lane1["name"], lane2["name"])
                if similarity_score > 0.7:  # Threshold for "similar" lanes
                    similar_lanes.append({
                        "lane1_id": lane1["id"],
                        "lane1_name": lane1["name"],
                        "lane2_id": lane2["id"],
                        "lane2_name": lane2["name"],
                        "similarity": similarity_score.item()
                    })

    lanes_analysis = {
        "common_lanes": list(common_lanes),
        "unique_to_bpmn1_lanes": list(unique_to_bpmn1_lanes),
        "unique_to_bpmn2_lanes": list(unique_to_bpmn2_lanes),
        "similar_lanes": similar_lanes  # Include similar lanes
    }

    # Gateways analysis
    gateways_bpmn1_ids = {gateway["id"] for gateway in gateways1}
    gateways_bpmn2_ids = {gateway["id"] for gateway in gateways2}
    gateway_differences = gateways_bpmn1_ids.symmetric_difference(gateways_bpmn2_ids)

    gateways_analysis = {
        "types": list(gateways_bpmn1_ids | gateways_bpmn2_ids),
        "gateway_differences": list(gateway_differences),
        "common_gateways": list(gateways_bpmn1_ids & gateways_bpmn2_ids),
        "unique_to_bpmn1_gateways": list(gateways_bpmn1_ids - gateways_bpmn2_ids),
        "unique_to_bpmn2_gateways": list(gateways_bpmn2_ids - gateways_bpmn1_ids)
    }

    # Behavioral analysis (placeholder for flow similarity)
    behavioral_analysis = {
        "flow_similarity": "Placeholder for flow similarity analysis. This would require more detailed parsing of the BPMN structure."
    }

    return {
        "structural": structural_similarity,
        "tasks": tasks_analysis,
        "lanes": lanes_analysis,
        "gateways": gateways_analysis,
        "behavioral": behavioral_analysis
    }

# Perform the analysis
analysis_results = analyze_bpmn(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2, gateways_bpmn1, gateways_bpmn2)

# Prepare the response content
response_content = {
    'message': {
        'content': f"Semantic similarity: {semantic_similarity.item()}\n"
                   f"Structural Analysis: {analysis_results['structural']}\n"
                   f"Tasks Analysis: {analysis_results['tasks']}\n"
                   f"Lanes Analysis: {analysis_results['lanes']}\n"
                   f"Gateways Analysis: {analysis_results['gateways']}\n"
                   f"Behavioral Analysis: {analysis_results['behavioral']}\n"
                   f"Overall Analysis: The diagrams have a semantic similarity score of {semantic_similarity.item()}. "
                   f"While they share some common tasks, there are notable differences in the lanes and gateways used, "
                   f"with some lanes and tasks being semantically similar but not identical."
    }
}

# Print the response content
print(response_content['message']['content'])


# In[37]:


import xml.etree.ElementTree as ET
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-distilroberta-v1')

def parse_bpmn(file_path):
    tasks = []
    lanes = []
    gateways = []

    tree = ET.parse(file_path)
    root = tree.getroot()

    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    for task in root.findall('.//bpmn:task', namespaces):
        tasks.append({
            "id": task.get('id'),
            "name": task.get('name')
        })

    for lane in root.findall('.//bpmn:lane', namespaces):
        lanes.append({
            "id": lane.get('id'),
            "name": lane.get('name')
        })

    for gateway in root.findall('.//bpmn:gateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name')
        })

    return tasks, lanes, gateways

bpmn1_file_path = "Signavio_source_partA.bpmn"  # Sostituisci con il percorso del tuo file
bpmn2_file_path = "Signavio_source_partA false (1).bpmn"  # Sostituisci con il percorso del tuo file

tasks_bpmn1, lanes_bpmn1, gateways_bpmn1 = parse_bpmn(bpmn1_file_path)
tasks_bpmn2, lanes_bpmn2, gateways_bpmn2 = parse_bpmn(bpmn2_file_path)

with open(bpmn1_file_path, 'r') as file:
    bpmn1_content = file.read()
with open(bpmn2_file_path, 'r') as file:
    bpmn2_content = file.read()

embeddings = model.encode([bpmn1_content, bpmn2_content], convert_to_tensor=True)

semantic_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])


def calculate_similarity(str1, str2):
    return util.pytorch_cos_sim(model.encode(str1, convert_to_tensor=True), model.encode(str2, convert_to_tensor=True))

def analyze_bpmn(tasks1, tasks2, lanes1, lanes2, gateways1, gateways2):
    # Analisi dei task
    tasks_bpmn1_ids = {task["id"]: task["name"] for task in tasks1}
    tasks_bpmn2_ids = {task["id"]: task["name"] for task in tasks2}
    tasks_bpmn1_names = {task["name"] for task in tasks1 if task["name"]}
    tasks_bpmn2_names = {task["name"] for task in tasks2 if task["name"]}

    common_tasks = tasks_bpmn1_ids.keys() & tasks_bpmn2_ids.keys()
    unique_to_bpmn1 = tasks_bpmn1_ids.keys() - tasks_bpmn2_ids.keys()
    unique_to_bpmn2 = tasks_bpmn2_ids.keys() - tasks_bpmn1_ids.keys()

    # Compara i task simili (semantici ma non uguali)
    similar_tasks = []
    for task1 in tasks1:
        for task2 in tasks2:
            if task1["id"] != task2["id"]:  # Ignora i match esatti
                similarity_score = calculate_similarity(task1["name"], task2["name"])
                if similarity_score > 0.8:  # Soglia per "task simili"
                    similar_tasks.append({
                        "task1_id": task1["id"],
                        "task1_name": task1["name"],
                        "task2_id": task2["id"],
                        "task2_name": task2["name"],
                        "similarity": similarity_score.item()
                    })

    #lane
    lanes_bpmn1_ids = {lane["id"]: lane["name"] for lane in lanes1}
    lanes_bpmn2_ids = {lane["id"]: lane["name"] for lane in lanes2}

    common_lanes = lanes_bpmn1_ids.keys() & lanes_bpmn2_ids.keys()
    unique_to_bpmn1_lanes = lanes_bpmn1_ids.keys() - lanes_bpmn2_ids.keys()
    unique_to_bpmn2_lanes = lanes_bpmn2_ids.keys() - lanes_bpmn1_ids.keys()

    similar_lanes = []
    for lane1 in lanes1:
        for lane2 in lanes2:
            if lane1["id"] != lane2["id"]:  # Ignora i match esatti
                similarity_score = calculate_similarity(lane1["name"], lane2["name"])
                if similarity_score > 0.7:  # Soglia per "lane simili"
                    similar_lanes.append({
                        "lane1_id": lane1["id"],
                        "lane1_name": lane1["name"],
                        "lane2_id": lane2["id"],
                        "lane2_name": lane2["name"],
                        "similarity": similarity_score.item()
                    })

    # Analisi dei gateway
    gateways_bpmn1_ids = {gateway["id"]: gateway["name"] for gateway in gateways1}
    gateways_bpmn2_ids = {gateway["id"]: gateway["name"] for gateway in gateways2}
    gateway_differences = gateways_bpmn1_ids.keys() ^ gateways_bpmn2_ids.keys()

    # Restituiamo l'analisi finale
    return {
        "global_similarity": semantic_similarity.item(),
        "tasks": {
            "common_tasks": [{"id": task_id, "name": tasks_bpmn1_ids[task_id]} for task_id in common_tasks],
            "unique_to_bpmn1": [{"id": task_id, "name": tasks_bpmn1_ids[task_id]} for task_id in unique_to_bpmn1],
            "unique_to_bpmn2": [{"id": task_id, "name": tasks_bpmn2_ids[task_id]} for task_id in unique_to_bpmn2],
            "similar_tasks": similar_tasks  # Task simili
        },
        "lanes": {
            "common_lanes": [{"id": lane_id, "name": lanes_bpmn1_ids[lane_id]} for lane_id in common_lanes],
            "unique_to_bpmn1_lanes": [{"id": lane_id, "name": lanes_bpmn1_ids[lane_id]} for lane_id in unique_to_bpmn1_lanes],
            "unique_to_bpmn2_lanes": [{"id": lane_id, "name": lanes_bpmn2_ids[lane_id]} for lane_id in unique_to_bpmn2_lanes],
            "similar_lanes": similar_lanes  # Lane simili
        },
        "gateways": {
            "gateway_differences": [{"id": gateway_id, "name": gateways_bpmn1_ids.get(gateway_id, gateways_bpmn2_ids.get(gateway_id))} for gateway_id in gateway_differences]
        }
    }

# Esegui l'analisi
analysis_results = analyze_bpmn(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2, gateways_bpmn1, gateways_bpmn2)

# Prepariamo il contenuto della risposta
response_content = {
    'message': {
        'content': f"Global similarity: {analysis_results['global_similarity']}\n"
                   f"Tasks Analysis:\n"
                   f"  Common Tasks: {analysis_results['tasks']['common_tasks']}\n"
                   f"  Unique to BPMN1 Tasks: {analysis_results['tasks']['unique_to_bpmn1']}\n"
                   f"  Unique to BPMN2 Tasks: {analysis_results['tasks']['unique_to_bpmn2']}\n"
                   f"  Similar Tasks (semantic similarity > 0.7): {analysis_results['tasks']['similar_tasks']}\n"
                   f"Lanes Analysis:\n"
                   f"  Common Lanes: {analysis_results['lanes']['common_lanes']}\n"
                   f"  Unique to BPMN1 Lanes: {analysis_results['lanes']['unique_to_bpmn1_lanes']}\n"
                   f"  Unique to BPMN2 Lanes: {analysis_results['lanes']['unique_to_bpmn2_lanes']}\n"
                   f"  Similar Lanes (semantic similarity > 0.7): {analysis_results['lanes']['similar_lanes']}\n"
                   f"Gateways Analysis:\n"
                   f"  Gateway Differences: {analysis_results['gateways']['gateway_differences']}\n"
                   f"Overall Analysis: The diagrams have a global similarity score of {analysis_results['global_similarity']}. "
                   f"While they share common tasks, lanes, and gateways, notable differences exist in task names, lane names, and gateway configurations, "
                   f"with some entities being semantically similar but not identical."
    }
}

# Stampa il contenuto della risposta
print(response_content['message']['content'])


# In[9]:


import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET


model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Funzione per ottenere l'embedding di una frase
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Media degli embeddings di tutti i token
    return embeddings

def cosine_similarity_score(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity_score(embedding1, embedding2)


def parse_bpmn(file_path):
    tasks = []
    lanes = []
    
    # Parsing del file BPMN
    tree = ET.parse(file_path)
    root = tree.getroot()

    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei task
    for task in root.findall('.//bpmn:task', namespaces):
        task_id = task.get('id')
        task_name = task.get('name', task_id) 
        tasks.append({"id": task_id, "name": task_name})

    # Estrazione delle lane
    for lane in root.findall('.//bpmn:lane', namespaces):
        lane_id = lane.get('id')
        lane_name = lane.get('name', lane_id)  
        lanes.append({"id": lane_id, "name": lane_name})

    return tasks, lanes


def analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2):
    # Analisi dei task
    task_similarities = []
    for task1 in tasks_bpmn1:
        for task2 in tasks_bpmn2:
            similarity = calculate_similarity(task1['name'], task2['name'])
            if similarity > 0.7:  
                task_similarities.append({
                    "task1_id": task1['id'],
                    "task1_name": task1['name'],
                    "task2_id": task2['id'],
                    "task2_name": task2['name'],
                    "similarity": similarity
                })
    
    # Analisi delle lane
    lane_similarities = []
    for lane1 in lanes_bpmn1:
        for lane2 in lanes_bpmn2:
            similarity = calculate_similarity(lane1['name'], lane2['name'])
            if similarity > 0.7:  
                lane_similarities.append({
                    "lane1_id": lane1['id'],
                    "lane1_name": lane1['name'],
                    "lane2_id": lane2['id'],
                    "lane2_name": lane2['name'],
                    "similarity": similarity
                })
    
    # Risultato dell'analisi con similitudine globale
    global_similarity = calculate_similarity(' '.join([task['name'] for task in tasks_bpmn1]), 
                                             ' '.join([task['name'] for task in tasks_bpmn2]))

    return {
        "global_similarity": global_similarity,
        "task_similarities": task_similarities,
        "lane_similarities": lane_similarities
    }

bpmn1_file_path = "Signavio_source_partA.bpmn" 
bpmn2_file_path = "Signavio_source_partA false (1).bpmn"  

tasks_bpmn1, lanes_bpmn1 = parse_bpmn(bpmn1_file_path)
tasks_bpmn2, lanes_bpmn2 = parse_bpmn(bpmn2_file_path)

result = analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2)

response_content = {
    "Semantic Similarity": result["semantic_similarity"],
    "Task Similarities": result["task_similarities"],
    "Lane Similarities": result["lane_similarities"],
}

print(f"Global Similarity: {response_content['Global Similarity']:.4f}")
print("\nTask Similarities:")
for similarity in response_content["Task Similarities"]:
    print(f"{similarity['task1_name']} <=> {similarity['task2_name']}: Similarity = {similarity['similarity']:.4f}")

print("\nLane Similarities:")
for similarity in response_content["Lane Similarities"]:
    print(f"{similarity['lane1_name']} <=> {similarity['lane2_name']}: Similarity = {similarity['similarity']:.4f}")


# In[8]:


#xml corrispondenza
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


def parse_bpmn_element_to_xml(element, sequence_flows):
    bpmn_element = ET.Element("Element")

    if element.tag.endswith('startEvent'):
        bpmn_element.set("type", "Start")
    elif element.tag.endswith('endEvent'):
        bpmn_element.set("type", "End")
    elif element.tag.endswith('task'):
        bpmn_element.set("type", "Task")
        bpmn_element.set("name", element.attrib['name'])
        bpmn_element.set("id", element.attrib['id'])
    elif element.tag.endswith('exclusiveGateway'):
        bpmn_element.set("type", "ExclusiveGateway")  
        bpmn_element.set("condition", "if ...") 
        outgoing_tasks = [flow for flow in sequence_flows if flow.attrib['sourceRef'] == element.attrib['id']]
        for flow in outgoing_tasks:
            task_element = ET.SubElement(bpmn_element, "Task")
            task_element.set("id", flow.attrib['targetRef'])
    elif element.tag.endswith('parallelGateway'):
        bpmn_element.set("type", "ParallelGateway")
    elif element.tag.endswith('inclusiveGateway'):
        bpmn_element.set("type", "InclusiveGateway")
    else:
        return None

    return bpmn_element


# Funzione per convertire un intero file BPMN in XML
def bpmn_to_xml(bpmn_file):
    tree = ET.parse(bpmn_file)
    root = tree.getroot()

    bpmn_xml = ET.Element("Process")
    namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    sequence_flows = root.findall('.//bpmn:sequenceFlow', namespace)

    for element in root.iter():
        parsed_element = parse_bpmn_element_to_xml(element, sequence_flows)
        if parsed_element is not None:
            bpmn_xml.append(parsed_element)

    return minidom.parseString(ET.tostring(bpmn_xml, encoding='utf-8')).toprettyxml(indent="    ")


# Funzioni per calcolo della similarità semantica con BERT
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Media degli embeddings di tutti i token
    return embeddings

def cosine_similarity_score(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity_score(embedding1, embedding2)


# Parsing del file BPMN per task e lane
def parse_bpmn(file_path):
    tasks = []
    lanes = []

    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    for task in root.findall('.//bpmn:task', namespaces):
        task_id = task.get('id')
        task_name = task.get('name', task_id) 
        tasks.append({"id": task_id, "name": task_name})

    for lane in root.findall('.//bpmn:lane', namespaces):
        lane_id = lane.get('id')
        lane_name = lane.get('name', lane_id)  
        lanes.append({"id": lane_id, "name": lane_name})

    return tasks, lanes


# Analisi BPMN con similarità tra task e lane
def analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2):
    task_similarities = []
    for task1 in tasks_bpmn1:
        for task2 in tasks_bpmn2:
            similarity = calculate_similarity(task1['name'], task2['name'])
            if similarity > 0.7:  
                task_similarities.append({
                    "task1_id": task1['id'],
                    "task1_name": task1['name'],
                    "task2_id": task2['id'],
                    "task2_name": task2['name'],
                    "similarity": similarity
                })
    
    lane_similarities = []
    for lane1 in lanes_bpmn1:
        for lane2 in lanes_bpmn2:
            similarity = calculate_similarity(lane1['name'], lane2['name'])
            if similarity > 0.7:  
                lane_similarities.append({
                    "lane1_id": lane1['id'],
                    "lane1_name": lane1['name'],
                    "lane2_id": lane2['id'],
                    "lane2_name": lane2['name'],
                    "similarity": similarity
                })
    
    global_similarity = calculate_similarity(
        ' '.join([task['name'] for task in tasks_bpmn1]), 
        ' '.join([task['name'] for task in tasks_bpmn2])
    )

    return {
        "global_similarity": global_similarity,
        "task_similarities": task_similarities,
        "lane_similarities": lane_similarities
    }


# Funzione per confrontare file BPMN
def compare_bpmn_files(bpmn_file_1, bpmn_file_2):
    print("Converting files to XML...")
    process_1_xml = bpmn_to_xml(bpmn_file_1)
    process_2_xml = bpmn_to_xml(bpmn_file_2)

    print("Parsing BPMN files...")
    tasks_bpmn1, lanes_bpmn1 = parse_bpmn(bpmn_file_1)
    tasks_bpmn2, lanes_bpmn2 = parse_bpmn(bpmn_file_2)

    print("Analyzing semantic similarity...")
    result = analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2)

    print(f"Global Similarity: {result['global_similarity']:.4f}")
    print("\nTask Similarities:")
    for similarity in result["task_similarities"]:
        print(f"{similarity['task1_name']} <=> {similarity['task2_name']}: Similarity = {similarity['similarity']:.4f}")

    print("\nLane Similarities:")
    for similarity in result["lane_similarities"]:
        print(f"{similarity['lane1_name']} <=> {similarity['lane2_name']}: Similarity = {similarity['similarity']:.4f}")


# Esempio di utilizzo
bpmn_file_1 = "Signavio_source_partA false (1).bpmn"  
 
bpmn_file_2 = "Signavio_source_partA.bpmn"  
 

compare_bpmn_files(bpmn_file_1, bpmn_file_2)


# In[2]:


#json matching
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import json  # Aggiunto per generare il file JSON

model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Funzione per ottenere l'embedding di una frase
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Media degli embeddings di tutti i token
    return embeddings

def cosine_similarity_score(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity_score(embedding1, embedding2)


def parse_bpmn(file_path):
    tasks = []
    lanes = []
    
    # Parsing del file BPMN
    tree = ET.parse(file_path)
    root = tree.getroot()

    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    # Estrazione dei task
    for task in root.findall('.//bpmn:task', namespaces):
        task_id = task.get('id')
        task_name = task.get('name', task_id) 
        tasks.append({"id": task_id, "name": task_name})

    # Estrazione delle lane
    for lane in root.findall('.//bpmn:lane', namespaces):
        lane_id = lane.get('id')
        lane_name = lane.get('name', lane_id)  
        lanes.append({"id": lane_id, "name": lane_name})

    return tasks, lanes


def analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2):
    # Analisi dei task
    task_similarities = []
    for task1 in tasks_bpmn1:
        for task2 in tasks_bpmn2:
            similarity = calculate_similarity(task1['name'], task2['name'])
            if similarity > 0.7:  
                task_similarities.append({
                    "task1_id": task1['id'],
                    "task1_name": task1['name'],
                    "task2_id": task2['id'],
                    "task2_name": task2['name'],
                    "similarity": float(similarity)  # Convertito in float
                })
    
    lane_similarities = []
    for lane1 in lanes_bpmn1:
        for lane2 in lanes_bpmn2:
            similarity = calculate_similarity(lane1['name'], lane2['name'])
            if similarity > 0.7:  
                lane_similarities.append({
                    "lane1_id": lane1['id'],
                    "lane1_name": lane1['name'],
                    "lane2_id": lane2['id'],
                    "lane2_name": lane2['name'],
                    "similarity": float(similarity)  # Convertito in float
                })
    
    global_similarity = calculate_similarity(' '.join([task['name'] for task in tasks_bpmn1]), 
                                             ' '.join([task['name'] for task in tasks_bpmn2]))

    return {
        "global_similarity": float(global_similarity),  # Convertito in float
        "task_similarities": task_similarities,
        "lane_similarities": lane_similarities
    }

bpmn1_file_path = "Signavio_source_partA.bpmn" 
bpmn2_file_path = "Signavio_source_partA false (1).bpmn"  

tasks_bpmn1, lanes_bpmn1 = parse_bpmn(bpmn1_file_path)
tasks_bpmn2, lanes_bpmn2 = parse_bpmn(bpmn2_file_path)

result = analyze_bpmn_with_mbert(tasks_bpmn1, tasks_bpmn2, lanes_bpmn1, lanes_bpmn2)

response_content = {
    "Global Similarity": result["global_similarity"],
    "Task Similarities": result["task_similarities"],
    "Lane Similarities": result["lane_similarities"]
}

output_file = "bpmn_similarity_result.json"
with open(output_file, 'w', encoding='utf-8') as json_file:
    json.dump(response_content, json_file, ensure_ascii=False, indent=4)

print(f"Result saved in {output_file}")
#jsonend


# In[15]:


import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Funzione per ottenere le informazioni sui gateway e i flussi di sequenza
def parse_bpmn_gateways(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    gateways = []
    sequence_flows = []

    # Estrazione dei gateway
    for gateway in root.findall('.//bpmn:exclusiveGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "ExclusiveGateway (XOR)"
        })

    for gateway in root.findall('.//bpmn:parallelGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "ParallelGateway"
        })

    for gateway in root.findall('.//bpmn:inclusiveGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "InclusiveGateway (OR)"
        })

    # Estrazione dei flussi di sequenza
    for flow in root.findall('.//bpmn:sequenceFlow', namespaces):
        sequence_flows.append({
            "source": flow.get('sourceRef'),
            "target": flow.get('targetRef'),
            "condition": flow.get('conditionExpression')  # Se presente, per condizioni di flusso
        })

    return gateways, sequence_flows


# Funzione per calcolare la similarità semantica tra due stringhe utilizzando BERT
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Media degli embeddings di tutti i token
    return embeddings

def cosine_similarity_score(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity_score(embedding1, embedding2)


# Funzione per visualizzare i gateway e i flussi di sequenza
def print_gateways_and_flows(gateways, sequence_flows):
    print("Gateways and their Connections:")

    # Stampa le informazioni sui gateway
    for gateway in gateways:
        print(f"Gateway ID: {gateway['id']}, Name: {gateway['name']}, Type: {gateway['type']}")

        # Trova i flussi di sequenza che collegano questo gateway
        outgoing_flows = [flow for flow in sequence_flows if flow['source'] == gateway['id']]
        incoming_flows = [flow for flow in sequence_flows if flow['target'] == gateway['id']]

        if outgoing_flows:
            print(f"  Outgoing Flows: {[flow['target'] for flow in outgoing_flows]}")
        if incoming_flows:
            print(f"  Incoming Flows: {[flow['source'] for flow in incoming_flows]}")

        # Se ci sono condizioni nei flussi, stampale
        for flow in outgoing_flows:
            if flow.get('condition'):
                print(f"    Condition: {flow['condition']}")

        print()


# Funzione per confrontare i gateway tra due file BPMN
def compare_gateways(gateways_bpmn1, gateways_bpmn2):
    print("\nComparing Gateways between two BPMN files:")
    
    # Confronta i gateway uno per uno
    for gateway1 in gateways_bpmn1:
        for gateway2 in gateways_bpmn2:
            if gateway1['id'] == gateway2['id']:
                similarity_score = 1  # I gateway con lo stesso ID sono esattamente uguali
                print(f"Gateway {gateway1['id']} ({gateway1['name']}) is the same in both files.")
                print(f"Type: {gateway1['type']} vs {gateway2['type']}")
                continue

            # Puoi anche calcolare una similarità basata sul nome del gateway
            similarity_score = calculate_similarity(gateway1['name'], gateway2['name'])
            if similarity_score > 0.7:  # Soglia di similarità
                print(f"Similar Gateway found:")
                print(f"Gateway 1 - ID: {gateway1['id']}, Name: {gateway1['name']}, Type: {gateway1['type']}")
                print(f"Gateway 2 - ID: {gateway2['id']}, Name: {gateway2['name']}, Type: {gateway2['type']}")
                print(f"Similarity: {similarity_score:.4f}")


# Funzione principale per il confronto tra i file BPMN
def compare_bpmn_files(file_path1, file_path2):
    # Estrazione dei gateway e flussi
    gateways_bpmn1, sequence_flows_bpmn1 = parse_bpmn_gateways(file_path1)
    gateways_bpmn2, sequence_flows_bpmn2 = parse_bpmn_gateways(file_path2)

    # Stampa i gateway e le connessioni per il primo file BPMN
    print(f"\nGateways in {file_path1}:")
    print_gateways_and_flows(gateways_bpmn1, sequence_flows_bpmn1)

    # Stampa i gateway e le connessioni per il secondo file BPMN
    print(f"\nGateways in {file_path2}:")
    print_gateways_and_flows(gateways_bpmn2, sequence_flows_bpmn2)

    # Confronto tra i gateway
    compare_gateways(gateways_bpmn1, gateways_bpmn2)


# Esempio di utilizzo
file_path1 = "Signavio_source_partA.bpmn"  # Sostituisci con il tuo file BPMN
file_path2 = "Signavio_source_partA false (1).bpmn"  # Sostituisci con il tuo secondo file BPMN

compare_bpmn_files(file_path1, file_path2)


# In[16]:


import xml.etree.ElementTree as ET
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch


# Funzione per ottenere le informazioni sui gateway e i flussi di sequenza
def parse_bpmn_gateways(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespaces = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    gateways = []
    sequence_flows = []

    # Estrazione dei gateway
    for gateway in root.findall('.//bpmn:exclusiveGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "ExclusiveGateway (XOR)"
        })

    for gateway in root.findall('.//bpmn:parallelGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "ParallelGateway"
        })

    for gateway in root.findall('.//bpmn:inclusiveGateway', namespaces):
        gateways.append({
            "id": gateway.get('id'),
            "name": gateway.get('name', gateway.get('id')),
            "type": "InclusiveGateway (OR)"
        })

    # Estrazione dei flussi di sequenza
    for flow in root.findall('.//bpmn:sequenceFlow', namespaces):
        sequence_flows.append({
            "source": flow.get('sourceRef'),
            "target": flow.get('targetRef'),
            "condition": flow.get('conditionExpression')  # Se presente, per condizioni di flusso
        })

    return gateways, sequence_flows


# Funzione per calcolare la similarità semantica tra due stringhe utilizzando BERT
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Media degli embeddings di tutti i token
    return embeddings

def cosine_similarity_score(embedding1, embedding2):
    similarity = cosine_similarity(embedding1.numpy(), embedding2.numpy())
    return similarity[0][0]

def calculate_similarity(text1, text2):
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    return cosine_similarity_score(embedding1, embedding2)


# Funzione per visualizzare i gateway e i flussi di sequenza
def print_gateways_and_flows(gateways, sequence_flows):
    print("Gateways and their Connections:")

   
    for gateway in gateways:
        print(f"Gateway ID: {gateway['id']}, Name: {gateway['name']}, Type: {gateway['type']}")

        # Trova i flussi di sequenza che collegano questo gateway
        outgoing_flows = [flow for flow in sequence_flows if flow['source'] == gateway['id']]
        incoming_flows = [flow for flow in sequence_flows if flow['target'] == gateway['id']]

        if outgoing_flows:
            print(f"  Outgoing Flows: {[flow['target'] for flow in outgoing_flows]}")
        if incoming_flows:
            print(f"  Incoming Flows: {[flow['source'] for flow in incoming_flows]}")

        # Se ci sono condizioni nei flussi, stampale
        for flow in outgoing_flows:
            if flow.get('condition'):
                print(f"    Condition: {flow['condition']}")

        print()


# Funzione per confrontare i gateway tra due file BPMN
def compare_gateways(gateways_bpmn1, gateways_bpmn2):
    print("\nComparing Gateways between two BPMN files:")

    similar_gateways = []

    # Confronta i gateway uno per uno
    for gateway1 in gateways_bpmn1:
        for gateway2 in gateways_bpmn2:
            if gateway1['id'] == gateway2['id']:
                # Calcolare la similarità semantica tra i nomi dei gateway
                similarity_score = calculate_similarity(gateway1['name'], gateway2['name'])
                if similarity_score > 0.7:  # Soglia di similarità
                    similar_gateways.append({
                        "gateway1_id": gateway1['id'],
                        "gateway1_name": gateway1['name'],
                        "gateway1_type": gateway1['type'],
                        "gateway2_id": gateway2['id'],
                        "gateway2_name": gateway2['name'],
                        "gateway2_type": gateway2['type'],
                        "similarity": similarity_score
                    })
    
    # Stampa dei gateway simili
    for gateway_pair in similar_gateways:
        print(f"\nSimilar Gateway Found:")
        print(f"Gateway 1 - ID: {gateway_pair['gateway1_id']}, Name: {gateway_pair['gateway1_name']}, Type: {gateway_pair['gateway1_type']}")
        print(f"Gateway 2 - ID: {gateway_pair['gateway2_id']}, Name: {gateway_pair['gateway2_name']}, Type: {gateway_pair['gateway2_type']}")
        print(f"Similarity: {gateway_pair['similarity']:.4f}")


# Funzione principale per il confronto tra i file BPMN
def compare_bpmn_files(file_path1, file_path2):
    # Estrazione dei gateway e flussi
    gateways_bpmn1, sequence_flows_bpmn1 = parse_bpmn_gateways(file_path1)
    gateways_bpmn2, sequence_flows_bpmn2 = parse_bpmn_gateways(file_path2)

    # Stampa i gateway e le connessioni per il primo file BPMN
    print(f"\nGateways in {file_path1}:")
    print_gateways_and_flows(gateways_bpmn1, sequence_flows_bpmn1)

    # Stampa i gateway e le connessioni per il secondo file BPMN
    print(f"\nGateways in {file_path2}:")
    print_gateways_and_flows(gateways_bpmn2, sequence_flows_bpmn2)

    # Confronto tra i gateway
    compare_gateways(gateways_bpmn1, gateways_bpmn2)


# Esempio di utilizzo
file_path1 = "Signavio_source_partA.bpmn"  # Sostituisci con il tuo file BPMN
file_path2 = "Signavio_source_partA false (1).bpmn"  # Sostituisci con il tuo secondo file BPMN

compare_bpmn_files(file_path1, file_path2)


# In[12]:


def create_gold_standard(tasks_bpmn1, tasks_bpmn2):
    gold_standard_task_matches = []

    for task1 in tasks_bpmn1:
        for task2 in tasks_bpmn2:
            if task1['name'] == task2['name']:  
                gold_standard_task_matches.append({
                    "task1_name": task1['name'],
                    "task2_name": task2['name']
                })

    return gold_standard_task_matches

gold_standard_task_matches = create_gold_standard(tasks_bpmn1, tasks_bpmn2)

print("Gold Standard Task Matches:")
for match in gold_standard_task_matches:
    print(f"{match['task1_name']} <=> {match['task2_name']}")


# In[22]:



def calculate_precision_recall(gold_standard, similarities):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for match in similarities:
        task1_name = match['task1_name']
        task2_name = match['task2_name']
        similarity = match['similarity']
        
        
        if any(gold_match['task1_name'] == task1_name and gold_match['task2_name'] == task2_name for gold_match in gold_standard):
            if similarity > 0.85:  # Definito un threshold per la similarità
                true_positives += 1
        else:
            if similarity > 0.85:  # Se non è nel gold standard, è un falso positivo
                false_positives += 1

    # Calcolo i falsi negativi (corrispondenze corrette che non sono state trovate)
    for gold_match in gold_standard:
        task1_name = gold_match['task1_name']
        task2_name = gold_match['task2_name']
        if not any(sim['task1_name'] == task1_name and sim['task2_name'] == task2_name for sim in similarities):
            false_negatives += 1

    # Calcola precisione e recall
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    
    return precision, recall

# Usa il gold standard e le similarità calcolate per i task
precision, recall = calculate_precision_recall(gold_standard_task_matches, result["task_similarities"])

# Calcola F1-Score
f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")


#gold standard (un set di corrispondenze corrette tra i task dei due modelli BPMN) per confrontare i risultati del  sistema di matching.
#Ho analizzato i task estratti dai due modelli BPMN e li hai confrontati, creando un elenco di corrispondenze basato su una condizione semplice (ad esempio, se i nomi dei task corrispondono).

#Ho utilizzato precision, recall e F1-score per misurare la qualità delle corrispondenze suggerite dal sistema. Queste metriche ti hanno permesso di quantificare quanto il sistema fosse preciso e completo nel trovare corrispondenze corrette tra i task.
#Precision: quanto delle corrispondenze suggerite dalla metrica sono effettivamente corrette.
#Recall: quanto delle corrispondenze effettivamente corrette sono state trovate dalla metrica.
#F1-score: una combinazione di precisione e recall che ti dà un'idea dell'equilibrio tra le due.


# In[ ]:




