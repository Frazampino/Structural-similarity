#!/usr/bin/env python
# coding: utf-8

# In[4]:


import xml.etree.ElementTree as ET

def extract_bpmn_structure(file_path):
    """
    Extract the structure and required elements from a BPMN file.
    
    :param file_path: Path to the BPMN XML file.
    :return: A dictionary containing sets of elements for comparison.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        bpmn_namespace = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
        
        structure = {
            'start_events': set(),
            'end_events': set(),
            'tasks': set(),
            'gateways': set()
        }
        
        for event in root.findall('.//bpmn:startEvent', bpmn_namespace):
            structure['start_events'].add(event.attrib['id'])
        for event in root.findall('.//bpmn:endEvent', bpmn_namespace):
            structure['end_events'].add(event.attrib['id'])
        for task in root.findall('.//bpmn:task', bpmn_namespace):
            structure['tasks'].add(task.attrib['id'])
        for gateway in root.findall('.//bpmn:exclusiveGateway', bpmn_namespace):
            structure['gateways'].add(gateway.attrib['id'])
        
        return structure
    except ET.ParseError as e:
        print(f"Parse error in file {file_path}: {e}")
        return None

def validate_against_structure(file_path, expected_structure):
    """
    Validate if the BPMN file conforms to the expected structure.
    
    :param file_path: Path to the BPMN XML file.
    :param expected_structure: A dictionary containing sets of expected elements.
    :return: True if the file conforms, otherwise False.
    """
    structure = extract_bpmn_structure(file_path)
    
    if structure is None:
        return False
    
    # Check if the structure matches the expected structure
    is_valid = True
    for key in expected_structure:
        if not expected_structure[key].issubset(structure[key]):
            print(f"Error: The file {file_path} does not conform to the expected syntax{key}.")
            is_valid = False
    
    if is_valid:
        print(f"The file {file_path} conforms to the expected structure.")
    return is_valid

def compare_bpmn_syntax(file_path1, file_path2):
    """
    Compare the syntax of two BPMN files to check if the second conforms to the first.
    
    :param file_path1: Path to the reference BPMN XML file.
    :param file_path2: Path to the BPMN XML file to validate.
    :return: None
    """
    expected_structure = extract_bpmn_structure(file_path1)
    
    if expected_structure is None:
        print(f"Error extracting structure from the reference file {file_path1}.")
        return
    
    validate_against_structure(file_path2, expected_structure)

# Example usage
file_name1 = 'diagram (5).bpmn'  # The BPMN file that defines the expected structure
file_name2 = 'diagram (6).bpmn'       # The BPMN file to validate against the reference

compare_bpmn_syntax(file_name1, file_name2)


# In[ ]:




