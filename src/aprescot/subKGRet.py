import json
import os, json, hashlib, time
from typing import Any, Dict, List, Set, Tuple, Optional
from langchain_openai import ChatOpenAI
from src.aprescot.metaqa import MetaQAKnowledgeGraph
from src.aprescot.umls import UMLSKnowledgeGraph
from src.aprescot.wikidata import WikiDataKnowledgeGraph
from src.aprescot.prompting import (
    SEED_ENTITY_INSTRUCTIONS, 
    SEED_ENTITY_PROMPT, 
    # SEED_ENTITY_JSON_KEYS
)


CACHE_DIR = os.environ.get("SUBGRAPH_CACHE_DIR", ".subgraph_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_key(kg: str, question: str, depth: int, params: Dict[str, Any]) -> str:
    payload = {
        "kg": kg,
        "question": question.strip(),
        "depth": depth,
        "params": params,           # include knobs that affect retrieval
        "v": 1,                     # bump if schema changes
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _cache_path(key: str) -> str:
    # one file per entry
    return os.path.join(CACHE_DIR, f"{key}.json")

def load_subgraph_cache(kg: str, question: str, depth: int, params: Dict[str, Any]
                       ) -> Optional[Tuple[List[str], Set[str], List[Dict], List[str]]]:
    key = _cache_key(kg, question, depth, params)
    path = _cache_path(key)
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    seed_nodes = doc["seed_nodes"]
    nodes_set = set(doc["nodes_list"])
    edge_dict_list = doc["edge_dict_list"]
    edge_desc_list = doc["edge_descriptions"]

    return seed_nodes, nodes_set, edge_dict_list, edge_desc_list

def save_subgraph_cache(kg: str, question: str, depth: int, params: Dict[str, Any],
                        seed_nodes: List[str], nodes_set: Set[str],
                        edge_dict_list: List[Dict], edge_desc_list: List[str]) -> None:
    key = _cache_key(kg, question, depth, params)
    path = _cache_path(key)
    doc = {
        "params": params,
        "ts": time.time(),
        "seed_nodes": seed_nodes,
        "nodes_list": sorted(nodes_set),
        "edge_dict_list": edge_dict_list,
        "edge_descriptions": edge_desc_list,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)

def get_seed_entities(question: str, kg: str):
    seed_entities_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"} })
    
    messages = [
        ("system", SEED_ENTITY_INSTRUCTIONS[kg]),
        ("user", SEED_ENTITY_PROMPT.format(question)),
    ]
    
    response = seed_entities_llm.invoke(messages)
    response_json = json.loads(response.content)

    return response_json["seed entities"]

def retrieve_subgraph(question: str, kg: str, depth: int, use_srtk: bool):
    if use_srtk and kg == "wikidata":
        beam_size = 30
        per_pred_cap = 32
        total_cap_per_node = 256
        max_nodes = 350
        scorer_model = "drt/srtk-scorer"
        compare_to_hypothetical_answer = False
        retriever_params = {
            "use_srtk": True,
            "max_hops": depth,
            "beam_size": beam_size,
            "per_pred_cap": per_pred_cap,
            "total_cap_per_node": total_cap_per_node,
            "max_nodes": max_nodes,
            "scorer_model": scorer_model,
            "hypothetical_answer": compare_to_hypothetical_answer,
        }
        seed_labels, nodes_set, edge_dict_list, edge_descriptions = None, None, None, None
        cached = load_subgraph_cache(kg, question, depth, params=retriever_params)
        if cached:
            seed_labels, nodes_set, edge_dict_list, edge_descriptions = cached
        else:
            wikidata_qa = WikiDataKnowledgeGraph(scorer_model=scorer_model)
            # seed_entities_txt = get_seed_entities(question, kg)
            seed_entities_txt = ["Germany"]
            # seed_entities_txt = ["Jean Rochefort"]
            # seed_entities_txt = ["President of the United States", "Q362 — World War II"]

            print("Seed Entities:", seed_entities_txt)
            wikidata_seed_nodes = wikidata_qa.find_wikidata_entities(seed_entities_txt)
            print("Seed QIDs:", wikidata_seed_nodes)
            q_ids = [node[0] for node in wikidata_seed_nodes]

            seed_labels, nodes_set, edge_dict_list, edge_descriptions = wikidata_qa.retrieve_with_srtk_style(
                question,
                q_ids,
                max_hops=depth,         # try 3 for tougher queries (slower)
                beam_size=beam_size,       # more edges per hop = higher recall
                per_pred_cap=per_pred_cap,    # cap fanout per (s,p)
                total_cap_per_node=total_cap_per_node,
                max_nodes=max_nodes,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
                add_labels=True,
            )

            print("Seed Labels:", seed_labels)
            print("Nodes:", nodes_set)
            print("Edge Count:", len(edge_dict_list))
            print("Edges:", edge_dict_list)

            save_subgraph_cache(kg, question, depth, params=retriever_params,
                                seed_nodes=seed_labels, nodes_set=nodes_set,
                                edge_dict_list=edge_dict_list, edge_desc_list=edge_descriptions)

        return seed_labels, nodes_set, edge_dict_list, edge_descriptions
        
    elif kg == "wikidata" and not use_srtk:
        wikidata_qa = WikiDataKnowledgeGraph()
        seed_entities_txt = get_seed_entities(question, kg)
        wikidata_seed_nodes = wikidata_qa.find_wikidata_entities(seed_entities_txt)
        q_ids = [node[0] for node in wikidata_seed_nodes]
        seed_labels = [node[0] for node in wikidata_seed_nodes]
        nodes_set, edge_dict_list, edge_descriptions = wikidata_qa.extract_relevant_subgraph(q_ids)

        return seed_labels, nodes_set, edge_dict_list, edge_descriptions

    elif kg == "meta-qa":
        movies_qa = MetaQAKnowledgeGraph()
        seed_nodes = get_seed_entities(question, kg)
        print("Seed Nodes: ", seed_nodes)
    
        edge_dict_list, nodes_set = movies_qa.extract_surrounding_subgraph(seed_nodes, depth)
        # edge_dict_list, nodes_set = movies_qa.extract_relevant_subgraph(seed_nodes, question, depth)
        edge_descriptions = movies_qa.extract_subgraph_edge_descriptions(edge_dict_list)

        return seed_nodes, nodes_set, edge_dict_list, edge_descriptions

    elif kg == "umls":
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