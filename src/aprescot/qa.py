from src.aprescot.subKGRet import *
from src.aprescot.prompting import *
from src.aprescot.matching import *
from src.aprescot.cytoVis import *
from src.aprescot.parsing import *
from experiments.subgraph_retriever import *

from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from openai import OpenAI
import json
import time

SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='./hf_models')

RETRIEVAL_SETTINGS = [
    {"name": "BFS", "use_srtk": False, "use_hyde": False, "use_pasr": False},
    {"name": "SSR", "use_srtk": True,  "use_hyde": False, "use_pasr": False},
    {"name": "SSR+HyDE", "use_srtk": True, "use_hyde": True, "use_pasr": False},
    {"name": "SSR+PASR", "use_srtk": True, "use_hyde": False, "use_pasr": True},
    {"name": "SSR+HyDE+PASR", "use_srtk": True, "use_hyde": True, "use_pasr": True},
]

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

    
    # use_srtk, use_hyde, use_pasr = False, False, False          # BFS
    # use_srtk, use_hyde, use_pasr = True, False, False           # Plain Similarity
    # use_srtk, use_hyde, use_pasr = True, False, True            # Similarity + PASR
    use_srtk, use_hyde, use_pasr = True, True, False            # Similarity + Hypothetical Answer
    # use_srtk, use_hyde, use_pasr = True, True, True             # Similarity + Hypothetical Answer + PASR

    use_subgraph_cache = False

    seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list = None, None, None, None
    instruction_msg, prompt = None, None
    llm_response, llm_final_answers, llm_cot = None, [], []

    if is_experiment:
        questions = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10"]
        # questions = ["Q4"]
        for q in questions:
            print(f"\n\n\n\n==========================")
            print(f"Running experiments for {q}")
            print(f"==========================")
            instruction_msg, prompt, llm_response, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot = \
                run_experiments(
                config_path="experiments/config.json",
                results_csv="experiments/results/results_log.csv",
                question_id=q
            )
        return instruction_msg, prompt, llm_response, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot

    else:
        # seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, time_elapsed = \
        #     retrieve_demo_subgraph(question, kg, use_srtk=use_srtk, use_hyde=use_hyde, use_cache=use_subgraph_cache)

        seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, subgraph_retrieval_time_elapsed = \
            retrieve_demo_subgraph(
                question, 
                kg=kg,
                use_srtk=use_srtk,
                use_hyde=use_hyde,
                # use_pasr=use_pasr,
                use_cache=use_subgraph_cache,
                # graph_file="experiments/germany_subgraph_depth3.txt",
                # graph_file="kg/meta-qa-kb.txt",
            )
        
        instruction_msg, prompt = create_prompt(question, kg, rag, llm, subgraph_edge_desc_list, new_reasoning, extension=extension)
        llm_response, llm_final_answers, llm_cot = ask_llm(llm, instruction_msg, prompt, new_reasoning, extension)

        if extension:
            # llm_final_answers, llm_cot = parse_reasoning_output(llm_response)
            llm_final_answers, reasoning = parse_reasoning_output(llm_response)
            
            
            llm_cot = extract_facts_as_triples(reasoning)
            llm_cot_with_inference_call = parse_llm_response(llm_response, question, triples=True)
            print("WITH THE NEW METHOD: ")
            print("Final Answers:", llm_final_answers)
            print("Reasoning:\n", reasoning)
            print("Extracted Facts:")

            for fact in llm_cot:
                print(f" - {fact}")

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


def run_experiments(config_path: str, results_csv: str, question_id: str, get_ground_truth: bool = False):
    with open(config_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Find the matching question
    question_conf = next((q for q in questions if q["id"] == question_id), None)
    if not question_conf:
        print(f"ERROR: Question ID '{question_id}' not found in {config_path}")
        return

    q = question_conf
    print(f"\n==========================")
    print(f"Running experiments for {q['id']} — {q['question']}")
    print(f"==========================")

    is_directed_kg = (q["kg"] != "meta-qa")

    results = []

    for setting in RETRIEVAL_SETTINGS:
        algo_name = setting["name"]
        print(f"\n----- Algorithm: {algo_name} -----")

        if get_ground_truth:
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, subgraph_time = \
                load_ground_truth_subgraph(q["ground_truth"])
        else:
            params = {
                "depth": q["params"].get("depth"),
                "beam_size": q["params"].get("beam_size"),
                "max_nodes": q["params"].get("max_nodes"),
            }
            seed_nodes, nodes_set, edge_dict_list, subgraph_edge_desc_list, subgraph_time = \
                retrieve_experiment_subgraph(
                    q["question"],
                    seed_entities=q["seeds"],
                    kg_name=q["kg"],
                    params=params,
                    use_srtk=setting["use_srtk"],
                    use_hyde=setting["use_hyde"],
                    use_pasr=setting["use_pasr"],
                    graph_file=q["graph_file"]
                )

        # --- Load LLM answers ---
        llm_final_answers, llm_cot = get_experiment_llm_answers(q["llm_answers"])

        # --- Matching phase ---
        start_match = time.perf_counter()
        node_to_answer_match, node_to_answer_id = match_nodes_using_embeddings(nodes_set, llm_final_answers[:1])
        matched_cot_list, edge_to_cot_match = match_edges(subgraph_edge_desc_list, llm_cot[:1], cot_in_triples=True)
        match_time = time.perf_counter() - start_match

        # --- Evaluate metrics ---
        precision, recall, f1 = evaluate_subgraph_extraction(
            gt_file=q["ground_truth"],
            pred_edges=edge_dict_list,
            undirected=not is_directed_kg
        )
        subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, node_to_answer_id)


        print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
        print(f"Subgraph Retrieval Time={subgraph_time:.2f}s  Matching Time={match_time:.2f}s")

        res = {
            "question_id": q["id"],
            "algorithm": algo_name,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "subgraph_time": round(subgraph_time, 3),
            "matching_time": round(match_time, 3),
            "total_edges": len(edge_dict_list),
            "total_nodes": len(nodes_set)
        }
        print("Result:", res)
        results.append(res)

    # --- Save or append results ---
    df = pd.DataFrame(results)
    try:
        existing = pd.read_csv(results_csv)
        df = pd.concat([existing, df], ignore_index=True)
    except FileNotFoundError:
        pass
    df.to_csv(results_csv, index=False)

    print(f"\nFinished all algorithms for {q['id']}. Results appended to {results_csv}.")
    return None, None, None, subgraph_edge_desc_list, node_to_answer_match, matched_cot_list, subgraph_elements_list, llm_final_answers, llm_cot
