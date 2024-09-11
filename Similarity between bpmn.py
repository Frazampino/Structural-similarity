#!/usr/bin/env python
# coding: utf-8

# In[25]:


from bpmn_python.bpmn_diagram_rep import BpmnDiagramGraph


# In[1]:



    def calculate_similarity(self, file_name1, file_name2):
        try:
            raw_bpmn = BpmnDiagramGraph()
            raw_bpmn.load_diagram_from_xml_file(file_name1)
            raw_nodes, start_end_node = self.get_bpmn_nodes(raw_bpmn)
            raw_edges = self.get_bpmn_flows(raw_bpmn, start_end_node)

            bpmn = BpmnDiagramGraph()
            bpmn.load_diagram_from_xml_file(file_name2)
            nodes, start_end_node = self.get_bpmn_nodes(bpmn)
            edges = self.get_bpmn_flows(bpmn, start_end_node)
            print(nodes)
            print(raw_nodes)
            if len(nodes) > 0 and len(raw_nodes) > 0:
           
                node_matching_sim, equivalence_mapping = self.calculate_node_matching_similarity(raw_nodes,
                                                                                                 nodes)
                print("node matching similarity: between", file_name1, "and", file_name2, ":",
                      round(node_matching_sim, 3))

                structure_sim = self.calculate_structure_similarity(raw_edges, raw_nodes, edges, nodes,
                                                                    equivalence_mapping)
                return node_matching_sim, structure_sim
            else:
                return 0, 0
        except BaseException as e:
            print("error in calculate similarity: ", e)
            return  0, 0


# In[2]:


similarity_calculator = BpmnSimilarity()
file_name1 = 'diagram (5).bpmn'  # Path to first BPMN file
file_name2 = 'diagram (6).bpmn'  # Path to second BPMN file

# Call the calculate_similarity method and print the result
node_sim, struct_sim = similarity_calculator.calculate_similarity(file1, file2)

print(f"Node similarity: {node_sim}")
print(f"Structural similarity: {struct_sim}")


# In[8]:


def get_bpmn_nodes(self, bpmn):
    try:
        # Assume we want to return a list of nodes and start/end node info.
        nodes = bpmn.get_nodes()  # Method should return a list of nodes
        start_end_node = {'start': None, 'end': None}
        
        # Logic to identify start and end nodes from the node list
        for node in nodes:
            if 'start' in node.lower():
                start_end_node['start'] = node
            elif 'end' in node.lower():
                start_end_node['end'] = node

        return nodes, start_end_node
    except Exception as e:
        print(f"Error in get_bpmn_nodes: {e}")
        return [], {}  # Return empty list and dictionary in case of error


# In[9]:


# Step 1: Define the class if not already defined
class BpmnSimilarity:
    def get_bpmn_nodes(self, bpmn):
        # Implementation of get_bpmn_nodes
        pass

    def get_bpmn_flows(self, bpmn, start_end_node):
        # Implementation of get_bpmn_flows
        pass

    def calculate_node_matching_similarity(self, raw_nodes, nodes):
        # Implementation of calculate_node_matching_similarity
        pass

    def calculate_structure_similarity(self, raw_edges, raw_nodes, edges, nodes, equivalence_mapping):
        # Implementation of calculate_structure_similarity
        pass

    def calculate_similarity(self, file_name1, file_name2):
        try:
            raw_bpmn = BpmnDiagramGraph()
            raw_bpmn.load_diagram_from_xml_file(file_name1)
            raw_nodes, start_end_node = self.get_bpmn_nodes(raw_bpmn)
            raw_edges = self.get_bpmn_flows(raw_bpmn, start_end_node)

            bpmn = BpmnDiagramGraph()
            bpmn.load_diagram_from_xml_file(file_name2)
            nodes, start_end_node = self.get_bpmn_nodes(bpmn)
            edges = self.get_bpmn_flows(bpmn, start_end_node)

            # Debug output to verify loaded nodes
            print(f"Nodes from {file_name1}:", raw_nodes)
            print(f"Nodes from {file_name2}:", nodes)

            if len(nodes) > 0 and len(raw_nodes) > 0:
                # Calculate node similarity
                node_matching_sim, equivalence_mapping = self.calculate_node_matching_similarity(raw_nodes, nodes)
                print(f"Node matching similarity between {file_name1} and {file_name2}: {round(node_matching_sim, 3)}")

                # Calculate structural similarity
                structure_sim = self.calculate_structure_similarity(raw_edges, raw_nodes, edges, nodes, equivalence_mapping)
                print(f"Structural similarity: {round(structure_sim, 3)}")

                return node_matching_sim, structure_sim
            else:
                print("No nodes found in one or both BPMN diagrams.")
                return 0, 0
        except Exception as e:
            print("Error in calculating similarity: ", e)
            return 0, 0

# Step 2: Create an instance of the class
similarity_calculator = BpmnSimilarity()

# Step 3: Define the file paths of the BPMN diagrams
file_name1 = 'diagram (5).bpmn'  # Path to first BPMN file
file_name2 = 'diagram (6).bpmn'

# Step 4: Call the calculate_similarity function
node_similarity, structural_similarity = similarity_calculator.calculate_similarity(file_name1, file_name2)

# Step 5: Print the results
print(f"Node Similarity: {node_similarity}")
print(f"Structural Similarity: {structural_similarity}")


# In[12]:


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



# In[ ]:


#BpmnDiagramGraph: Carica i diagrammi BPMN e individua i nodi (task, eventi, ecc.) e i flussi di sequenza (edges).
#BpmnSimilarity:
#get_bpmn_nodes: Recupera i nodi dal diagramma BPMN.
#get_bpmn_flows: Recupera i flussi di sequenza (edges) dal diagramma BPMN.
#calculate_structure_similarity: Confronta i flussi di sequenza dei due diagrammi BPMN. Viene calcolata come rapporto tra gli archi corrispondenti tra i due diagrammi rispetto al numero massimo di archi presenti in uno dei due diagrammi.
#calculate_similarity: Carica i due diagrammi BPMN dai file XML specificati, estrae i nodi e i flussi, e calcola direttamente la similarità strutturale.

