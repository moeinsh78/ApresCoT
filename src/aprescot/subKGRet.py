import json
from langchain_openai import ChatOpenAI
from src.aprescot.graph import UMLSKnowledgeGraph, MetaQAKnowledgeGraph
from src.aprescot.prompting import (
    SEED_ENTITY_INSTRUCTIONS, 
    SEED_ENTITY_PROMPT, 
    # SEED_ENTITY_JSON_KEYS
)


def get_seed_entities(question: str, kg: str):
    seed_entities_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"} })
    
    messages = [
        ("system", SEED_ENTITY_INSTRUCTIONS[kg]),
        ("user", SEED_ENTITY_PROMPT.format(question)),
    ]
    
    response = seed_entities_llm.invoke(messages)
    response_json = json.loads(response.content)

    return response_json["seed entities"]


def retrieve_uc2_subgraph(question: str, kg: str):
    seed_nodes = ["Fungus"]
    nodes_set = set(["Fungus", "Mental or Behavioral Dysfunction", "Cell or Molecular Dysfunction", "Virus", "Eukaryote", "Enzyme", "Hormone", "Experimental Model of Disease",
                     "Mammal", "Mental Process", "Neoplastic Process", "Reptile", "Bird", "Organism", "Pathologic Function", "Bacterium", "Family Group", "Injury or Poisoning",
                     "Vitamin", "Occupation or Discipline", "Chemical", "Organism Function", "Cell Component", "Patient or Disabled Group", "Professional or Occupational Group",
                     "Acquired Abnormality", "Anatomical Abnormality", "Congenital Abnormality"])

    edge_dict_list = [
        {"from": "Fungus", "to": "Mental or Behavioral Dysfunction", "label": "causes", "description": "Fungus causes Mental or Behavioral Dysfunction."},
        {"from": "Fungus", "to": "Eukaryote", "label": "isa", "description": "Fungus is a Eukaryote."},
        {"from": "Fungus", "to": "Experimental Model of Disease", "label": "causes", "description": "Fungus causes Experimental Model of Disease."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Mammal", "label": "affects", "description": "Mental or Behavioral Dysfunction affects Mammal."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Mental Process", "label": "affects", "description": "Mental or Behavioral Dysfunction affects Mental Process."},
        {"from": "Experimental Model of Disease", "to": "Injury or Poisoning", "label": "occurs_in", "description": "Experimental Model of Disease occurs in Injury or Poisoning."},
        {"from": "Fungus", "to": "Enzyme", "label": "location_of", "description": "Fungus is the location of Enzyme."},
        {"from": "Fungus", "to": "Hormone", "label": "location_of", "description": "Fungus is the location of Hormone."},
        {"from": "Experimental Model of Disease", "to": "Patient or Disabled Group", "label": "occurs_in", "description": "Experimental Model of Disease occurs in Patient or Disabled Group."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Neoplastic Process", "label": "affects", "description": "Mental or Behavioral Dysfunction affects Neoplastic Process."},
        {"from": "Fungus", "to": "Cell or Molecular Dysfunction", "label": "causes", "description": "Fungus causes Cell or Molecular Dysfunction."},
        {"from": "Enzyme", "to": "Chemical", "label": "isa", "description": "Enzyme is a Chemical."},
        {"from": "Hormone", "to": "Cell Component", "label": "disrupts", "description": "Hormone disrupts Cell Component."},
        {"from": "Cell or Molecular Dysfunction", "to": "Bird", "label": "affects", "description": "Cell or Molecular Dysfunction affects Bird."},
        {"from": "Experimental Model of Disease", "to": "Family Group", "label": "occurs_in", "description": "Experimental Model of Disease occurs in Family Group."},
        {"from": "Experimental Model of Disease", "to": "Professional or Occupational Group", "label": "occurs_in", "description": "Experimental Model of Disease occurs in Professional or Occupational Group."},
        {"from": "Hormone", "to": "Pathologic Function", "label": "causes", "description": "Hormone causes Pathologic Function."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Acquired Abnormality", "label": "complicates", "description": "Mental or Behavioral Dysfunction complicates Acquired Abnormality."},
        {"from": "Cell or Molecular Dysfunction", "to": "Organism", "label": "affects", "description": "Cell or Molecular Dysfunction affects Organism."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Congenital Abnormality", "label": "complicates", "description": "Mental or Behavioral Dysfunction complicates Congenital Abnormality."},
        {"from": "Fungus", "to": "Virus", "label": "interacts_with", "description": "Fungus interacts with Virus."},
        {"from": "Cell or Molecular Dysfunction", "to": "Pathologic Function", "label": "affects", "description": "Cell or Molecular Dysfunction affects Pathologic Function."},
        {"from": "Virus", "to": "Bacterium", "label": "interacts_with", "description": "Virus interacts with Bacterium."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Reptile", "label": "affects", "description": "Mental or Behavioral Dysfunction affects Reptile."},
        {"from": "Cell or Molecular Dysfunction", "to": "Mammal", "label": "affects", "description": "Cell or Molecular Dysfunction affects Mammal."},
        {"from": "Virus", "to": "Occupation or Discipline", "label": "issue_in", "description": "Virus issue in Occupation or Discipline."},
        {"from": "Eukaryote", "to": "Organism", "label": "isa", "description": "Eukaryote is a Organism."},
        {"from": "Enzyme", "to": "Organism Function", "label": "complicates", "description": "Enzyme complicates Organism Function."},
        {"from": "Cell or Molecular Dysfunction", "to": "Mental Process", "label": "affects", "description": "Cell or Molecular Dysfunction affects Mental Process."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Anatomical Abnormality", "label": "complicates", "description": "Mental or Behavioral Dysfunction complicates Anatomical Abnormality."},
        {"from": "Mental or Behavioral Dysfunction", "to": "Cell or Molecular Dysfunction", "label": "complicates", "description": "Mental or Behavioral Dysfunction complicates Cell or Molecular Dysfunction."},
        {"from": "Enzyme", "to": "Vitamin", "label": "interacts_with", "description": "Enzyme interacts with Vitamin."},
    ]
    
    edge_descriptions = [edge_dict["description"] for edge_dict in edge_dict_list]

    return seed_nodes, nodes_set, edge_dict_list, edge_descriptions


def retrieve_subgraph(question: str, kg: str, depth: int):
    if kg == "meta-qa":
        movies_qa = MetaQAKnowledgeGraph()
        seed_nodes = get_seed_entities(question, kg)
        print("Seed Nodes: ", seed_nodes)
    
        edge_dict_list, nodes_set = movies_qa.extract_surrounding_subgraph(seed_nodes, depth)
        # edge_dict_list, nodes_set = movies_qa.extract_relevant_subgraph(seed_nodes, question, depth)
        edge_descriptions = movies_qa.extract_subgraph_edge_descriptions(edge_dict_list)

        return seed_nodes, nodes_set, edge_dict_list, edge_descriptions

    if kg == "umls":
        umls_qa = UMLSKnowledgeGraph()
        seed_nodes = get_seed_entities(question, kg)
        print("Seed Nodes: ", seed_nodes)

        # edge_dict_list, nodes_set = umls_qa.extract_surrounding_subgraph(seed_nodes, depth)
        edge_dict_list, nodes_set = umls_qa.extract_relevant_subgraph(seed_nodes, question, depth)
        edge_descriptions = umls_qa.extract_subgraph_edge_descriptions(edge_dict_list)

        return seed_nodes, nodes_set, edge_dict_list, edge_descriptions

    else:
        print("Invalid Knowledge Graph:", kg)
        return None, None, None