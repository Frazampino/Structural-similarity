

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

def structural_similarity(score, min_value=0, max_value=5):
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




normalized_similarity = normalize_similarity(raw_similarity, min_possible_value, max_possible_value)
print(f"Structural Similarity: {normalized_similarity:.3f}")


# In[ ]:





# In[ ]:




