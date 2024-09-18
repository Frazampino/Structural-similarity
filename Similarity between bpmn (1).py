#!/usr/bin/env python
# coding: utf-8

# In[25]:


from bpmn_python.bpmn_diagram_rep import BpmnDiagramGraph


# In[5]:


#CORRECT METHOD
import xml.etree.ElementTree as ET

class BpmnDiagramGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def load_diagram_from_xml_file(self, file_name):
        # Parse the XML BPMN file
        tree = ET.parse(file_name)
        root = tree.getroot()

        # Namespace for BPMN (assumed based on standard BPMN XML structure)
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

        # Extract nodes (tasks, events, etc.) based on BPMN standard tags
        for task in root.findall('.//bpmn:task', ns):
            self.nodes.append(task.attrib.get('id'))
        
        for event in root.findall('.//bpmn:startEvent', ns):
            self.nodes.append(event.attrib.get('id'))
        
        for event in root.findall('.//bpmn:endEvent', ns):
            self.nodes.append(event.attrib.get('id'))

        # Extract sequence flows (edges) between nodes
        for flow in root.findall('.//bpmn:sequenceFlow', ns):
            source = flow.attrib.get('sourceRef')
            target = flow.attrib.get('targetRef')
            self.edges.append((source, target))

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges


class BpmnSimilarity:
    def get_bpmn_nodes(self, bpmn):
        # Retrieve nodes from the BPMN diagram
        return bpmn.get_nodes()

    def get_bpmn_flows(self, bpmn):
        # Retrieve edges from the BPMN diagram
        return bpmn.get_edges()

    def calculate_structure_similarity(self, raw_edges, edges):
        # Basic structure similarity based on the comparison of edges
        total_edges = max(len(raw_edges), len(edges))
        if total_edges == 0:
            return 0  # Avoid division by zero

        matching_edges = len(set(raw_edges) & set(edges))  # Intersection of edges
        structure_similarity = matching_edges / total_edges
        return structure_similarity

    def calculate_similarity(self, file_name1, file_name2):
        try:
            # Load the first BPMN diagram
            raw_bpmn = BpmnDiagramGraph()
            raw_bpmn.load_diagram_from_xml_file(file_name1)
            raw_edges = self.get_bpmn_flows(raw_bpmn)

            # Load the second BPMN diagram
            bpmn = BpmnDiagramGraph()
            bpmn.load_diagram_from_xml_file(file_name2)
            edges = self.get_bpmn_flows(bpmn)

            # Debug output to verify loaded nodes
            print(f"Edges from {file_name1}:", raw_edges)
            print(f"Edges from {file_name2}:", edges)

            # Calculate structural similarity
            structure_sim = self.calculate_structure_similarity(raw_edges, edges)
            print(f"Structural similarity: {round(structure_sim, 3)}")

            return structure_sim
        except Exception as e:
            print("Error in calculating similarity: ", e)
            return 0

# Step 2: Create an instance of the class
similarity_calculator = BpmnSimilarity()

# Step 3: Define the file paths of the BPMN diagrams
file_name1 = 'diagram (5).bpmn'  # Path to first BPMN file
file_name2 = 'diagram (6).bpmn'

# Step 4: Call the calculate_similarity function to get structural similarity directly
structural_similarity = similarity_calculator.calculate_similarity(file_name1, file_name2)



# In[6]:


#BpmnDiagramGraph: Carica i diagrammi BPMN e individua i nodi (task, eventi, ecc.) e i flussi di sequenza (edges).
#BpmnSimilarity:
#get_bpmn_nodes: Recupera i nodi dal diagramma BPMN.
#get_bpmn_flows: Recupera i flussi di sequenza (edges) dal diagramma BPMN.
#calculate_structure_similarity: Confronta i flussi di sequenza dei due diagrammi BPMN. Viene calcolata come rapporto tra gli archi corrispondenti tra i due diagrammi rispetto al numero massimo di archi presenti in uno dei due diagrammi.
#calculate_similarity: Carica i due diagrammi BPMN dai file XML specificati, estrae i nodi e i flussi, e calcola direttamente la similarità strutturale.


# In[ ]:





# In[14]:


#syntax checking between process models
import xml.etree.ElementTree as ET

# Funzione per estrarre i nodi principali di un BPMN (attività, eventi, gateway)
def extract_bpmn_elements(file_path):
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        print(f"Errore di parsing del file BPMN: {e}")
        return []
    
    root = tree.getroot()

   
    bpmn_namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    
    
    elements = []
    
    for event in root.findall('.//bpmn:startEvent', bpmn_namespace):
        elements.append(event.attrib['id'])
    for event in root.findall('.//bpmn:endEvent', bpmn_namespace):
        elements.append(event.attrib['id'])

 
    for task in root.findall('.//bpmn:task', bpmn_namespace):
        elements.append(task.attrib['id'])

    # Estrazione dei gateway esclusivi
    for gateway in root.findall('.//bpmn:exclusiveGateway', bpmn_namespace):
        elements.append(gateway.attrib['id'])

    return elements


def calculate_semantic_similarity(bpmn1, bpmn2):
    G1_elements = set(extract_bpmn_elements(bpmn1))  
    G2_elements = set(extract_bpmn_elements(bpmn2))  

   
    intersection_size = len(G1_elements.intersection(G2_elements))
 
    synonym_penalty = 0
    for s in G1_elements.difference(G2_elements):
        for l in G2_elements.difference(G1_elements):
            if s == l:  
                synonym_penalty += 1

    max_set_size = max(len(G1_elements), len(G2_elements))

    
    sem_similarity = 1.0 * intersection_size + 0.75 * synonym_penalty
    normalized_sem_similarity = sem_similarity / max_set_size

    return normalized_sem_similarity

# Esempio di utilizzo
bpmn1 = 'diagram (5).bpmn'  
bpmn2 = 'diagram (6).bpmn'  

# Calcolo della similarità
similarity = calculate_semantic_similarity(bpmn1, bpmn2)
print("Normalized SEM Similarity:", similarity)


# In[4]:


import xml.etree.ElementTree as ET


def extract_bpmn_nodes_and_edges(file_path):
    try:
        tree = ET.parse(file_path)
    except ET.ParseError as e:
        print(f"Errore di parsing del file BPMN: {e}")
        return [], []
    
    root = tree.getroot()
    bpmn_namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

    nodes = []
    edges = []

    for event in root.findall('.//bpmn:startEvent', bpmn_namespace):
        nodes.append(event.attrib['id'])
    for event in root.findall('.//bpmn:endEvent', bpmn_namespace):
        nodes.append(event.attrib['id'])
    for task in root.findall('.//bpmn:task', bpmn_namespace):
        nodes.append(task.attrib['id'])
    for gateway in root.findall('.//bpmn:exclusiveGateway', bpmn_namespace):
        nodes.append(gateway.attrib['id'])

    # Estrazione delle sequenze di flusso (edges)
    for flow in root.findall('.//bpmn:sequenceFlow', bpmn_namespace):
        source = flow.attrib.get('sourceRef')
        target = flow.attrib.get('targetRef')
        edges.append((source, target))

    return nodes, edges


def check_deadlocks(file_path):
    nodes, edges = extract_bpmn_nodes_and_edges(file_path)

    if not nodes or not edges:
        return "Error: No node in the BPMN."

    reachable = set()
    for edge in edges:
        source, target = edge
        reachable.add(target)
    
   
    deadlocked_nodes = set(nodes) - reachable
    if deadlocked_nodes:
        return f"Deadlock in the nodes: {deadlocked_nodes}"
    
    return "No deadlock"


bpmn_file = 'diagram (5).bpmn'  # Sostituisci con il tuo file BPMN

result = check_deadlocks(bpmn_file)
print(result)


# In[10]:


import xml.etree.ElementTree as ET

class BpmnDiagram:
    def __init__(self, file_name):
        self.nodes = set()
        self.edges = set()
        self.load_diagram(file_name)

    def load_diagram(self, file_name):
        tree = ET.parse(file_name)
        root = tree.getroot()
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        
        for task in root.findall('.//bpmn:task', ns):
            self.nodes.add(task.attrib['id'])
        
        for event in root.findall('.//bpmn:startEvent', ns):
            self.nodes.add(event.attrib['id'])
        
        for event in root.findall('.//bpmn:endEvent', ns):
            self.nodes.add(event.attrib['id'])
        
        for flow in root.findall('.//bpmn:sequenceFlow', ns):
            self.edges.add((flow.attrib['sourceRef'], flow.attrib['targetRef']))

def normalize_similarity(score, min_value=0, max_value=5):
    """
    Normalize the similarity score to be within the range [0, 1].
    
    :param score: The similarity score to normalize.
    :param min_value: The minimum possible score.
    :param max_value: The maximum possible score.
    :return: Normalized similarity score.
    """
    normalized_score = (score - min_value) / (max_value - min_value)
    # Ensure the normalized score is within [0, 1]
    normalized_score = max(0, min(1, normalized_score))
    return normalized_score

# Example usage
raw_similarity = 4.333
min_possible_value = 0  # Minimum value for similarity score
max_possible_value = 5  # Example maximum value (could be adjusted based on your context)

normalized_similarity = normalize_similarity(raw_similarity, min_possible_value, max_possible_value)
print(f"Structural Similarity: {normalized_similarity:.3f}")


# In[ ]:





# In[ ]:




