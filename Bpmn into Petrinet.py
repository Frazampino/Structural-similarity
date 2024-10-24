

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

# BPMN conversion in PetriNet
def bpmn_to_petri_net(tasks, events, gateways):
    net = PetriNet()

    
    for event in events:
        net.add_place(event[1])

   
    for task in tasks:
        net.add_transition(task)

   
    for gateway in gateways:
        net.add_transition(gateway[1])

   
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

# Petrinet in PNML
class PNMLExporter:
    def __init__(self, petri_net):
        self.petri_net = petri_net

    def export(self, file_path):
        pnml = ET.Element("pnml")
        net = ET.SubElement(pnml, "net", id="net1", type="http://www.pnml.org/version-2009/grammar/pnml")

       
        for place in self.petri_net.places:
            place_elem = ET.SubElement(net, "place", id=place)
            name_elem = ET.SubElement(place_elem, "name")
            ET.SubElement(name_elem, "text").text = place

      
        for transition in self.petri_net.transitions:
            trans_elem = ET.SubElement(net, "transition", id=transition)
            name_elem = ET.SubElement(trans_elem, "name")
            ET.SubElement(name_elem, "text").text = transition

        for i, (source, target) in enumerate(self.petri_net.arcs):
            arc_elem = ET.SubElement(net, "arc", id=f"a{i+1}", source=source, target=target)

        # Scrivi l'albero XML nel file PNML
        tree = ET.ElementTree(pnml)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)

        print(f"PNML file exported to: {file_path}")


if __name__ == "__main__":
   
    bpmn_file_path = "diagram (5).bpmn"  # Inserisci il percorso del tuo file BPMN

    
    tasks, events, gateways = parse_bpmn(bpmn_file_path)

   
    petri_net = bpmn_to_petri_net(tasks, events, gateways)

    
    petri_net.display()

    
    pnml_file_path = "petri_net_output.pnml"  # Inserisci il nome del file PNML di output
    pnml_exporter = PNMLExporter(petri_net)
    pnml_exporter.export(pnml_file_path)


# In[ ]:




