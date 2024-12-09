from typing import List, Dict, Sequence
from rage.retriever import Source
from src.aprescot.subKGRet import retrieve_subgraph, retrieve_uc2_subgraph
from src.aprescot.prompting import create_prompt
from src.aprescot.matching import match_edges, match_nodes
from src.aprescot.cytoVis import build_cyto_subgraph_elements_list
from langchain_openai import ChatOpenAI
import json



def ask_llm(llm: str, instruction_msg: str, prompt: str):
    qa_llm = ChatOpenAI(model=llm, temperature=0, model_kwargs={"response_format": {"type": "json_object"} })
    messages = [
        ("system", instruction_msg),
        ("user", prompt),
    ]
    response = qa_llm.invoke(messages)
    response_json = json.loads(response.content)

    answer_list = [str(answer) for answer in response_json["Answer"]]
    cot_list = [str(cot) for cot in response_json["Chain of Thought"]]

    return response.content, answer_list, cot_list


def perform_qa(llm: str, kg: str, question: str, rag: bool):
    # Subgraph Retrieval
    # seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = retrieve_subgraph(question, kg, depth = 2)
    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = None, None, None, None
    if kg == "umls":
        seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = retrieve_uc2_subgraph(question, "umls")
    else:
        seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = retrieve_subgraph(question, kg, depth = 2)

    # LLM QA Process
    instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list)
    llm_response, llm_final_answers, llm_cot = ask_llm(llm, instruction_msg, prompt)

    # Matcher
    node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
    matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot)

    # Visualizations
    subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, node_to_answer_id)

    return instruction_msg, prompt, llm_response, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list
