from typing import List, Dict, Sequence
from src.aprescot.subKGRet import retrieve_subgraph, retrieve_uc2_subgraph
from src.aprescot.prompting import create_prompt
from src.aprescot.matching import match_edges, match_nodes
from src.aprescot.cytoVis import build_cyto_subgraph_elements_list
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json


def parse_llm_response(response: str, question: str, triples: bool = False) -> Dict[str, List[str]]:
    qa_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"} })
    if triples:
        messages = [
        ("system", TRIPLES_PARSE_RESPONSE_INSTRUCTION),
        ("user", 'Question: \n{}\n\nModel Response: \n{}\nA model has responded in this way to the given question. Please parse out and decompose the reasoning steps of this model into independent atomic facts shaped in knowledge triples, and identify the single items of the final answer that the model has provided. Shape them all in a JSON format.\nThis would be an example of knowledge triple: <Paris, capital_of, France>'.format(question, response)),
    ]
    else: 
        messages = [
            ("system", PARSE_RESPONSE_INSTRUCTION),
            ("user", 'Question: \n{}\n\nModel Response: \n{}\nA model has responded in this way to the given question. Please parse out and decompose the reasoning steps of this model into independent atomic facts shaped in knowledge triples, and identify the single items of the final answer that the model has provided. Shape them all in a JSON format.\n'.format(question, response)),
        ]
    response = qa_llm.invoke(messages)
    response_json = json.loads(response.content)

    answer_list = [str(answer) for answer in response_json["answer"]]
    reasoning_list = [str(cot) for cot in response_json["reasoning"]]

    print("Parsed Response:", response_json)
    return answer_list, reasoning_list
    


def ask_llm(llm: str, instruction_msg: str, prompt: str, new_reasoning: bool = True):
    json_formatting = not new_reasoning
    if json_formatting:
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

    else:
        client = OpenAI()

        response = client.responses.create(
            model=llm,
            instructions=instruction_msg,
            input=prompt,
        )
        return response.output_text, [], []


def perform_qa(llm: str, kg: str, question: str, rag: bool):
    new_reasoning = True
    # Subgraph Retrieval
    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = None, None, None, None
    instruction_msg, prompt = None, None
    llm_response, llm_final_answers, llm_cot = None, [], []

    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = retrieve_subgraph(question, kg, depth=2, use_srtk=True)

    instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning)
    llm_response, llm_final_answers, llm_cot = ask_llm(llm, instruction_msg, prompt, new_reasoning)

    if new_reasoning:
        llm_final_answers, llm_cot = parse_llm_response(llm_response, question)

    # Matcher
    node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
    matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot)

    print("Done matching and subgraph")
    print("Seed Nodes:", seed_nodes)
    print("Nodes:", nodes_set)
    print("Edges:")
    for edge in edge_dict_list:
        print(edge)
    
    print("Edge to CoT Match:")
    for match in edge_to_cot_match.items():
        print(match)

    print("Node to Answer ID Match:")
    for match in node_to_answer_id.items():
        print(match)

    # Visualizations
    subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, node_to_answer_id)

    print("Subgraph Elements List:")
    for element in subgraph_elements_list:
        print(element)

    return instruction_msg, prompt, llm_response, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot



TRIPLES_PARSE_RESPONSE_INSTRUCTION = \
"""
A question has been asked from a model, and the model has provided its reasoning process and final answers to the question in plain text without any specific structure.
Your task is to parse the response, extract the reasoning and answers, structure the reasoning sentences in shape of knowledge triples (<subject, predicate, object>), and output them in a structured JSON format. 
The reasoning steps should be a collection of atomic facts, and then you should try to extract knowledge triples as if they are knowledge graph edges. 
Each reasoning step should be a clear statement that contributes to the final answer and only one piece of information. Decompose the thought process into independent sentences, each including a single fact or piece of reasoning, and then shape the triples.
Also the final answers should be a list of the answers that the model has provided in response to the question.
You will be given the question that was asked and the model's response.
Please extract the reasoning steps and final answers from the response and return them in a JSON object with two keys:
1. "reasoning": a list of atomic reasoning steps used to answer the question, in shape of knowledge triples (<subject, predicate, object>)
2. "answer": a list of final answers to the question.
"""


PARSE_RESPONSE_INSTRUCTION = \
"""
A question has been asked from a model, and the model has provided its reasoning process and final answers to the question in plain text without any specific structure.
Your task is to parse the response, extract the reasoning and answers, and output them in a structured JSON format. 
The reasoning steps should be a collection of atomic facts. Each reasoning step should be a clear statement that contributes to the final answer and only one piece of information. 
Decompose the thought process into independent sentences, each including a single fact or piece of reasoning. You should decompose the reasoning into atomic sentences.
Also the final answers should be a list of the answers that the model has provided in response to the question.
You will be given the question that was asked and the model's response.
Please extract the reasoning steps and final answers from the response and return them in a JSON object with two keys:
1. "reasoning": a list of atomic reasoning steps used to answer the question.
2. "answer": a list of final answers to the question.
"""
