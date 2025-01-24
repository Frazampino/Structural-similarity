#!/usr/bin/env python
# coding: utf-8
#json matching
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
import json  # Aggiunto per generare il file JSON

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


def parse_bpmn(file_path):
    tasks = []
    lanes = []
    
    # Parsing del file BPMN
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





 








