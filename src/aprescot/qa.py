from src.aprescot.subKGRet import retrieve_demo_subgraph, retrieve_experiment_subgraph, retrieve_uc2_subgraph
from src.aprescot.prompting import create_prompt
from src.aprescot.matching import match_edges, match_nodes, match_nodes_using_embeddings
from src.aprescot.cytoVis import build_cyto_subgraph_elements_list
from src.aprescot.parsing import parse_llm_response, concat_triple, parse_reasoning_output
from src.aprescot.experiments import (
    evaluate_subgraph_extraction, 
    get_nodes_and_edges_matching_gt, 
    load_ground_truth_subgraph,
    get_experiment_llm_answers,
)
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
import json
import time

def ask_llm(llm: str, instruction_msg: str, prompt: str, new_reasoning: bool = True, extension: bool = False):
    if extension:
        json_formatting = False
    elif new_reasoning:
        json_formatting = True

    if json_formatting:
        qa_llm = ChatOpenAI(
            model=llm,
            temperature=0,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        messages = [
            SystemMessage(content=instruction_msg),
            HumanMessage(content=prompt),
        ]

        response = qa_llm.invoke(messages)

        try:
            response_json = json.loads(response.content)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON returned from LLM:\n" + response.content)

        answer_list = [str(a) for a in response_json.get("Answers", [])]
        cot_list = [str(c) for c in response_json.get("Chain of Thought", [])]

        return response.content, answer_list, cot_list

    else:
        client = OpenAI()
        if llm == "o3-mini":
            response = client.responses.create(
                model=llm,
                input=[{
                    "role": "user",
                    "content": prompt,
                }],
                reasoning={"effort": "medium", "summary": "auto"},
                store=False,
            )
            answer_text = response.output_text
            reasoning_summary = response.reasoning.summary
            reasoning_tok = response.usage.output_tokens_details.reasoning_tokens

            output = "Reasoning Tokens: " + str(reasoning_tok) + "\n\n" + "Reasoning: \n" + reasoning_summary + "\n\n" + "Answer: \n" + answer_text + "\n\n"
            return output, [], []

        else:
            response = client.responses.create(
                temperature=0,
                model=llm,
                instructions=instruction_msg,
                input=prompt,
            )
            return response.output_text, [], []

def perform_qa(llm: str, kg: str, question: str, rag: bool):
    new_reasoning = False           # New reasoning format is not bound to CoT and JSON formatting
    extension = True                # Whether to use the extended prompt with context from the retrieved subgraph

    parse_to_triples = False        # Indicates whether to parse reasoning to triples since it affects matching too

    is_experiment = True            # Whether the code is running for the purpose of experimenting and benchmarking 
    get_ground_truth = False        # Whether to get ground-truth edges and answers for evaluation and matching
    ground_truth_file_dir = "ground_truth/germany.txt"
    llm_answers_file_dir = "llm_answers/germany.txt"
    is_directed_kg = (kg != "meta-qa")

    
    # use_srtk, use_hyde, use_pasr = False, False, False          # BFS
    use_srtk, use_hyde, use_pasr = True, False, False           # Plain Similarity
    # use_srtk, use_hyde, use_pasr = True, False, True            # Similarity + PASR
    # use_srtk, use_hyde, use_pasr = True, True, False            # Similarity + Hypothetical Answer
    # use_srtk, use_hyde, use_pasr = True, True, True             # Similarity + Hypothetical Answer + PASR

    use_subgraph_cache = False

    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = None, None, None, None
    instruction_msg, prompt = None, None
    llm_response, llm_final_answers, llm_cot = None, [], []
    precision, recall, f1 = 0, 0, 0

    if is_experiment:
        if get_ground_truth:
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, subgraph_retrieval_time_elapsed = load_ground_truth_subgraph(ground_truth_file_dir)
        else:
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, subgraph_retrieval_time_elapsed = \
                retrieve_experiment_subgraph(
                    question, 
                    kg_name=kg,
                    use_srtk=use_srtk,
                    use_hyde=use_hyde,
                    use_pasr=use_pasr,
                    graph_file="experiments/germany_subgraph_depth3.txt",
                    # graph_file="kg/meta-qa-kb.txt",
                )

        # llm_final_answers, llm_cot = get_nodes_and_edges_matching_gt(gt_file=ground_truth_file_dir, pred_edges=edge_dict_list, directed=is_directed_kg)
        llm_final_answers, llm_cot = get_experiment_llm_answers(answers_file=llm_answers_file_dir)

        # llm_final_answers = []
        # llm_cot = []
        instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning, extension=False)
        
        # node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
        start = time.perf_counter()
        node_to_answer_match, node_to_answer_id = match_nodes_using_embeddings(nodes_set, llm_final_answers[:1])
        matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot[:1], cot_in_triples=parse_to_triples)
        end = time.perf_counter()

        matching_time_elapsed = end - start
        subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, node_to_answer_id)

        precision, recall, f1 = evaluate_subgraph_extraction(
            gt_file=ground_truth_file_dir,
            pred_edges=edge_dict_list,
            undirected=True
        )

        print(f"\nPrecision= {precision:.3f}, Recall= {recall:.3f}, F1= {f1:.3f}")
        print(f"\nSubgraph Retrieval Time: {subgraph_retrieval_time_elapsed:.2f} sec")
        print(f"\nMatching Time: {matching_time_elapsed:.2f} sec")

        return instruction_msg, prompt, None, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot

    
    else:
        seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, time_elapsed = \
            retrieve_demo_subgraph(question, kg, use_srtk=use_srtk, use_hyde=use_hyde, use_cache=use_subgraph_cache)

        instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning, extension=extension)
        llm_response, llm_final_answers, llm_cot = ask_llm(llm, instruction_msg, prompt, new_reasoning, extension)

        if extension:
            llm_final_answers, llm_cot = parse_reasoning_output(llm_response)

        if new_reasoning:
            llm_final_answers, llm_cot = parse_llm_response(llm_response, question, triples=parse_to_triples)

        if parse_to_triples:
            llm_cot = [concat_triple(cot_step) for cot_step in llm_cot]

        # Matcher
        # node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
        node_to_answer_match, node_to_answer_id = match_nodes_using_embeddings(nodes_set, llm_final_answers)
        matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot, cot_in_triples=parse_to_triples)

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
