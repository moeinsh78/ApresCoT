import json
import os, json, hashlib, time
from typing import Any, Dict, List, Set, Tuple, Optional
from langchain_openai import ChatOpenAI
from src.aprescot.metaqa import MetaQAKnowledgeGraph
from src.aprescot.umls import UMLSKnowledgeGraph
from src.aprescot.wikidata import WikiDataKnowledgeGraph
from src.aprescot.experiments import ExperimentSubgraphRetriever
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
        "params": params,
        "v": 1,
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


def retrieve_experiment_subgraph(question: str, kg_name: str, use_srtk: bool, use_hyde: bool = False, use_cc: bool = False, graph_file: str = None):
    ##########################################################
    ################## Retrieval Parameters ##################
    scorer_model = "sentence-transformers/all-MiniLM-L6-v2"
    depth = 3
    beam_size = 16
    max_nodes = 500
    compare_to_hypothetical_answer = use_hyde
    ##########################################################

    experiment_retriever = ExperimentSubgraphRetriever(kg_name=kg_name, kg_directory=graph_file, scorer_model=scorer_model)
    
    start = time.perf_counter()
    seed_entities = get_seed_entities(question, kg_name)

    if not use_srtk:
        edge_dict_list = experiment_retriever.get_bfs_subgraph(seed_entities, depth=depth)
        end = time.perf_counter()

        edge_descriptions = experiment_retriever.extract_subgraph_edge_descriptions(edge_dict_list)
        nodes_set = get_nodes_set(edge_dict_list)
    else:
        if use_cc:
            edge_dict_list, nodes_set = experiment_retriever.extract_with_srtk_cumulative_context(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )

        else:
            edge_dict_list, nodes_set = experiment_retriever.extract_with_srtk(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )
        end = time.perf_counter()
        edge_descriptions = extract_subgraph_edge_descriptions(edge_dict_list)
        
    return seed_entities, nodes_set, edge_dict_list, edge_descriptions, end - start

def get_nodes_set(edge_dict_list: List[Dict]):
    nodes_set = set()
    for edge in edge_dict_list:
        nodes_set.add(edge["from"])
        nodes_set.add(edge["to"])
    return nodes_set


def extract_subgraph_edge_descriptions(edge_dict_list):
    edge_desc_list = []

    for edge_dict in edge_dict_list:    
        edge_desc_list.append(edge_dict["description"])

    return edge_desc_list


def retrieve_demo_subgraph(question: str, kg: str, use_srtk: bool, use_hyde: bool = False, use_cache: bool = True):
    depth = 2
    compare_to_hypothetical_answer = use_hyde
    seed_entities = get_seed_entities(question, kg)
    
    match kg:
        case "wikidata":
            if use_srtk:
                beam_size = 16
                per_pred_cap = 32
                total_cap_per_node = 256
                max_nodes = 500
                # scorer_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
                scorer_model = "sentence-transformers/all-MiniLM-L6-v2"
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
                if use_cache:
                    start = time.perf_counter()
                    cached = load_subgraph_cache(kg, question, depth, params=retriever_params)
                    seed_labels, nodes_set, edge_dict_list, edge_descriptions = cached
                    end = time.perf_counter()
                else:
                    wikidata_qa = WikiDataKnowledgeGraph(scorer_model=scorer_model, use_local_db=True)
                    # seed_entities = ["Germany"]
                    # seed_entities = ["Jean Rochefort"]
                    # seed_entities = ["President of the United States", "Q362 — World War II"]

                    print("Seed Entities:", seed_entities)
                    wikidata_seed_nodes = wikidata_qa.find_wikidata_entities(seed_entities)
                    print("Seed QIDs:", wikidata_seed_nodes)
                    seed_qids = [node[0] for node in wikidata_seed_nodes]

                    start = time.perf_counter()

                    seed_labels, nodes_set, edge_dict_list, edge_descriptions = wikidata_qa.retrieve_with_srtk_style(
                        question, seed_qids,
                        max_hops=depth, beam_size=beam_size, per_pred_cap=per_pred_cap,
                        total_cap_per_node=total_cap_per_node, max_nodes=max_nodes,
                        compare_to_hypothetical_answer=compare_to_hypothetical_answer,
                        add_labels=True,
                    )
                    end = time.perf_counter()
                    if use_cache:
                        save_subgraph_cache(kg, question, depth, params=retriever_params,
                                            seed_nodes=seed_labels, nodes_set=nodes_set,
                                            edge_dict_list=edge_dict_list, edge_desc_list=edge_descriptions)

                return seed_labels, nodes_set, edge_dict_list, edge_descriptions, end - start
            else:
                pass
                # Implement BFS for wikidata
                # wikidata_qa = WikiDataKnowledgeGraph()
                # wikidata_seed_nodes = wikidata_qa.find_wikidata_entities(seed_entities)
                # q_ids = [node[0] for node in wikidata_seed_nodes]
                # seed_labels = [node[0] for node in wikidata_seed_nodes]
                
                # start = time.perf_counter()
                # nodes_set, edge_dict_list, edge_descriptions = wikidata_qa.extract_relevant_subgraph(q_ids)
                # end = time.perf_counter()

                # return seed_labels, nodes_set, edge_dict_list, edge_descriptions, start - end
        case "meta-qa":
            movies_qa = MetaQAKnowledgeGraph()
            print("Seed Entities: ", seed_entities)
            
            start = time.perf_counter()

            if use_srtk:
                edge_dict_list, nodes_set = movies_qa.extract_relevant_subgraph_srtk(
                    seed_entities, 
                    question, 
                    max_hops=depth, 
                    beam_size=beam_size, 
                    max_nodes=max_nodes,
                    compare_to_hypothetical_answer=compare_to_hypothetical_answer,
                )
            else:
                edge_dict_list, nodes_set = movies_qa.extract_surrounding_subgraph(seed_entities, depth)
            
            end = time.perf_counter()

            edge_descriptions = movies_qa.extract_subgraph_edge_descriptions(edge_dict_list)

            return seed_entities, nodes_set, edge_dict_list, edge_descriptions, end - start

        case "umls":
            umls_qa = UMLSKnowledgeGraph()
            print("Seed Entities: ", seed_entities)

            # edge_dict_list, nodes_set = umls_qa.extract_surrounding_subgraph(seed_nodes, depth)
            
            start = time.perf_counter()
            edge_dict_list, nodes_set = umls_qa.extract_relevant_subgraph(seed_entities, question, depth)
            end = time.perf_counter()

            edge_descriptions = umls_qa.extract_subgraph_edge_descriptions(edge_dict_list)

            return seed_entities, nodes_set, edge_dict_list, edge_descriptions, end - start

        case _:
            print("Invalid Knowledge Graph:", kg)
            return None, None, None, None, None




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