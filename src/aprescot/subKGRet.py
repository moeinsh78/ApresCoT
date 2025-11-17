import json
import os, json, hashlib, time
from typing import Any, Dict, List, Set, Tuple, Optional
from langchain_openai import ChatOpenAI
from src.aprescot.metaqa import MetaQAKnowledgeGraph
from src.aprescot.umls import UMLSKnowledgeGraph
from src.aprescot.wikidata import WikiDataKnowledgeGraph
from experiments.subgraph_retriever import ExperimentSubgraphRetriever
from src.aprescot.demoGraphs import DemoSubgraphRetriever
from src.aprescot.prompting import (
    SEED_ENTITY_INSTRUCTIONS, 
    SEED_ENTITY_PROMPT, 
)

from openai import OpenAI


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
HF_MODELS_DIR = os.path.join(PROJECT_ROOT, 'hf_models')

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


def retrieve_experiment_subgraph(question: str, seed_entities : str, hypothetical_answer: str, kg_name: str, params: Dict[str, Any], use_srtk: bool, use_hyde: bool = False, use_pasr: bool = False, graph_file: str = None):
    ##########################################################
    ################## Retrieval Parameters ##################
    scorer_model = 'sentence-transformers/all-MiniLM-L6-v2'
    depth = params.get("depth")
    beam_size = params.get("beam_size")
    max_nodes = params.get("max_nodes")
    compare_to_hypothetical_answer = use_hyde
    #########################################################

    experiment_retriever = ExperimentSubgraphRetriever(kg_name=kg_name, kg_directory=graph_file, scorer_model=scorer_model, model_cache_folder=HF_MODELS_DIR)
    
    start = time.perf_counter()

    if not use_srtk:
        edge_dict_list, nodes_set = experiment_retriever.get_bfs_subgraph(seed_entities, depth=depth, expand_ending_nodes=False)
        end = time.perf_counter()

        edge_descriptions = extract_subgraph_edge_descriptions(edge_dict_list)
    else:
        if use_pasr:
            edge_dict_list, nodes_set = experiment_retriever.extract_with_srtk_cumulative_context(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                hypothetical_answer=hypothetical_answer,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )

        else:
            edge_dict_list, nodes_set = experiment_retriever.extract_with_srtk(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                hypothetical_answer=hypothetical_answer,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )
        end = time.perf_counter()
        edge_descriptions = extract_subgraph_edge_descriptions(edge_dict_list)
        
    return seed_entities, nodes_set, edge_dict_list, edge_descriptions, end - start


def extract_subgraph_edge_descriptions(edge_dict_list):
    edge_desc_list = []

    for edge_dict in edge_dict_list:    
        edge_desc_list.append(edge_dict["description"])

    return edge_desc_list


def retrieve_demo_subgraph_cached(question: str, kg_name: str, use_srtk: bool, use_hyde: bool = True, use_pasr: bool = True):
    seed_entities = get_seed_entities(question, kg_name)
    
    ##########################################################
    ################## Retrieval Parameters ##################
    scorer_model = 'sentence-transformers/all-MiniLM-L6-v2'
    depth = 3
    beam_size = 32
    total_cap_per_node = 256
    max_nodes = 2000
    compare_to_hypothetical_answer = use_hyde
    #########################################################

    hypothetical_answer = ""
    if use_hyde:
        hypothetical_answer = generate_hypothetical_answer(question)

    demo_retriever = DemoSubgraphRetriever(kg_name=kg_name, scorer_model=scorer_model, model_cache_folder=HF_MODELS_DIR)
    
    start = time.perf_counter()

    if not use_srtk:
        edge_dict_list, nodes_set = demo_retriever.get_bfs_subgraph(seed_entities, depth=depth, expand_ending_nodes=False)
        end = time.perf_counter()

        edge_descriptions = extract_subgraph_edge_descriptions(edge_dict_list)
    else:
        if use_pasr:
            edge_dict_list, nodes_set = demo_retriever.extract_with_srtk_cumulative_context(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                hypothetical_answer=hypothetical_answer,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )

        else:
            edge_dict_list, nodes_set = demo_retriever.extract_with_srtk(
                seed_entities, 
                question, 
                max_hops=depth, 
                beam_size=beam_size, 
                max_nodes=max_nodes,
                hypothetical_answer=hypothetical_answer,
                compare_to_hypothetical_answer=compare_to_hypothetical_answer,
            )
        end = time.perf_counter()
        edge_descriptions = extract_subgraph_edge_descriptions(edge_dict_list)
        
    return seed_entities, nodes_set, edge_dict_list, edge_descriptions, end - start


def retrieve_demo_subgraph(question: str, kg: str, use_srtk: bool, use_hyde: bool = False, use_cache: bool = True):
    depth = 2
    beam_size = 32
    per_pred_cap = 32
    total_cap_per_node = 256
    max_nodes = 2000
    scorer_model = "sentence-transformers/all-MiniLM-L6-v2"
    # scorer_model = os.path.join(HF_MODELS_DIR, 'sentence-transformers_all-MiniLM-L6-v2')

    compare_to_hypothetical_answer = use_hyde
    seed_entities = get_seed_entities(question, kg)
    
    match kg:
        case "wikidata":
            if use_srtk:
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
                    # seed_entities = ["Alexander Fleming"]
                    # seed_entities = ["Albert Einstein"]
                    # seed_entities = ["Arjen Robben"]
                    # seed_entities = ["Julián Alvarezz", "Ángel Di María"]
                    seed_entities = ["Thierry Henry", "Eden Hazard"]

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


# This function is only used to generate a hypothetical answer for experiments,
# so that we can pass the same answer to all retrieval algorithms.
def generate_hypothetical_answer(question: str, model_name="gpt-4o-mini", temperature=0, max_tokens=512, n=1) -> str:
    client = OpenAI()
    result = client.chat.completions.create(
        messages=[{"role":"user", "content": HYPOTHETICAL_ANSWER_PROMPT.format(question)}],
        model=model_name, 
        max_completion_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    return result.choices[0].message.content


HYPOTHETICAL_ANSWER_PROMPT = """Please write a passage to answer the question.
Question: {}
Passage:"""