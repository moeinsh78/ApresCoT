from src.aprescot.subKGRet import retrieve_subgraph, retrieve_uc2_subgraph
from src.aprescot.prompting import create_prompt
from src.aprescot.matching import match_edges, match_nodes
from src.aprescot.cytoVis import build_cyto_subgraph_elements_list
from src.aprescot.parsing import parse_llm_response, concat_triple
from src.aprescot.experiments import evaluate_subgraph_extraction, get_nodes_and_edges_matching_gt, load_ground_truth_subgraph
from langchain_openai import ChatOpenAI
from openai import OpenAI
import json
import time


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
                model=llm,
                instructions=instruction_msg,
                input=prompt,
            )
            return response.output_text, [], []

def perform_qa(llm: str, kg: str, question: str, rag: bool):
    is_experiment = True         # Whether the code is running for the purpose of experimenting and benchmarking 

    new_reasoning = True            # New reasoning format is not bound to CoT and JSON formatting

    parse_to_triples = True         # Indicates whether to parse reasoning to triples since it affects matching too

    use_srtk = True
    use_hyde = False
    depth = 2

    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = None, None, None, None
    instruction_msg, prompt = None, None
    llm_response, llm_final_answers, llm_cot = None, [], []
    precision, recall, f1 = 0, 0, 0

    if is_experiment:
        ground_truth_file_dir = "ground_truth/shawshank.txt"
        get_ground_truth = True   # Whether to get ground-truth edges and answers for evaluation and matching

        if get_ground_truth:
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, time_elapsed = load_ground_truth_subgraph(ground_truth_file_dir)
        else:
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, time_elapsed = retrieve_subgraph(question, kg, depth=depth, is_experiment_setup=is_experiment, use_srtk=use_srtk, hyde=use_hyde)

        llm_final_answers, llm_cot = get_nodes_and_edges_matching_gt(gt_file=ground_truth_file_dir, pred_edges=edge_dict_list, undirected=True)

        llm_final_answers = []
        llm_cot = []
        instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning)
        
        node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
        matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot, cot_in_triples=parse_to_triples)

        subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, node_to_answer_id)

        precision, recall, f1 = evaluate_subgraph_extraction(
            gt_file=ground_truth_file_dir,
            pred_edges=edge_dict_list,
            undirected=True
        )

        minutes = int(time_elapsed // 60)
        seconds = time_elapsed % 60

        print(f"\nPrecision= {precision:.3f}, Recall= {recall:.3f}, F1= {f1:.3f}")
        print(f"\nSubgraph Retrieval Execution Time: {minutes} min {seconds:.2f} sec")

        return instruction_msg, prompt, None, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot

    
    else:
        seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, time_elapsed = retrieve_subgraph(question, kg, depth=depth, is_experiment_setup=is_experiment, use_srtk=use_srtk, hyde=use_hyde)

        instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning)
        llm_response, llm_final_answers, llm_cot = ask_llm(llm, instruction_msg, prompt, new_reasoning)

        if new_reasoning:
            llm_final_answers, llm_cot = parse_llm_response(llm_response, question, triples=parse_to_triples)

        if parse_to_triples:
            llm_cot = [concat_triple(cot_step) for cot_step in llm_cot]

        # Matcher
        node_to_answer_match, node_to_answer_id = match_nodes(nodes_set, llm_final_answers)
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
