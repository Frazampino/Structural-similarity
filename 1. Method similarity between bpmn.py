#Method 1 (basic level): Compares the sequences of flows (edges) between nodes and calculates a similarity based on the intersection of these flows.
#Simple and straightforward, it only considers the flows between nodes and provides a quick measure.
#It does not take into account the type of nodes (tasks, events, etc.), meaning that diagrams with the same structure but with different types of nodes may appear very similar, even if they are semantically different.

#!/usr/bin/env python
# coding: utf-8
import xml.etree.ElementTree as ET

class BpmnDiagramGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def load_diagram_from_xml_file(self, file_name):
        
        tree = ET.parse(file_name)
        root = tree.getroot()

       
        ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}

       
        for task in root.findall('.//bpmn:task', ns):
            self.nodes.append(task.attrib.get('id'))
        
        for event in root.findall('.//bpmn:startEvent', ns):
            self.nodes.append(event.attrib.get('id'))
        
        for event in root.findall('.//bpmn:endEvent', ns):
            self.nodes.append(event.attrib.get('id'))

        
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
            return 0  
        matching_edges = len(set(raw_edges) & set(edges))  # Intersection of edges
        structure_similarity = matching_edges / total_edges
        return structure_similarity

    def calculate_similarity(self, file_name1, file_name2):
        try:
           
            raw_bpmn = BpmnDiagramGraph()
            raw_bpmn.load_diagram_from_xml_file(file_name1)
            raw_edges = self.get_bpmn_flows(raw_bpmn)

          
            bpmn = BpmnDiagramGraph()
            bpmn.load_diagram_from_xml_file(file_name2)
            edges = self.get_bpmn_flows(bpmn)

     
            print(f"Edges from {file_name1}:", raw_edges)
            print(f"Edges from {file_name2}:", edges)

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


