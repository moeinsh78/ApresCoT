from typing import List, Dict, Tuple, Set
import numpy as np
import networkx as nx
from openai import OpenAI

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class ExperimentSubgraphRetriever:
    def __init__(self, kg_name: str, kg_directory: str, scorer_model: str, model_cache_folder: str):
        self.kg_name = kg_name
        self.graph = self.load_graph_from_file(edge_list_file=kg_directory)
        self.similarity_model = SentenceTransformer(
            scorer_model, model_cache_folder
        )
        self.ending_node_relations = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]


    def load_graph_from_file(self, edge_list_file: str) -> nx.MultiDiGraph | nx.MultiGraph:
        if self.kg_name == "meta-qa":
            G = nx.MultiGraph()
        elif self.kg_name in ["wikidata", "umls"]:
            G = nx.MultiDiGraph()

        with open(edge_list_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue  # skip comments or empty lines
                parts = line.split("|")
                if len(parts) != 3:
                    continue

                head, relation, tail = parts
                if self.kg_name == "meta-qa":
                    description = create_meta_qa_description(head, relation, tail)
                    G.add_edge(head, tail, label=relation, description=description)
                else:
                    description = f"{head} {relation} {tail}"
                G.add_edge(head, tail, label=relation, description=description)

        return G

    def get_bfs_subgraph(self, seed_entities: List[str], depth: int, expand_ending_nodes: bool = True) -> Tuple[List[Dict], Set[str]]:
        edge_dict_list = []
        to_be_expanded = list(seed_entities)
        visited = set()
        curr_depth = 0
        nodes = set(seed_entities)

        while curr_depth < depth:
            to_expand_count = len(to_be_expanded)
            print(f"\n[Depth {curr_depth}] Expanding {to_expand_count} nodes...")

            for _ in range(to_expand_count):
                curr_node = to_be_expanded.pop(0)
                if curr_node in visited:
                    continue

                visited.add(curr_node)
                neighbors = list(nx.bfs_edges(self.graph, curr_node, depth_limit=1))

                for src, dst in neighbors:
                    # Always record all edges (even if dst already visited)
                    for i in range(self.graph.number_of_edges(src, dst)):
                        edge = {
                            "from": src,
                            "to": dst,
                            "label": self.graph.edges[src, dst, i]["label"],
                            "description": self.graph.edges[src, dst, i]["description"],
                        }
                        edge_dict_list.append(edge)
                        nodes.add(dst)

                    # Only expand the node if it hasn't been expanded before
                    if dst not in visited:
                        if (expand_ending_nodes) or (
                            self.graph.edges[src, dst, 0]["label"] not in self.ending_node_relations
                        ):
                            to_be_expanded.append(dst)


            curr_depth += 1

        print(f"\n[INFO] BFS completed. Found {len(nodes)} nodes and {len(edge_dict_list)} edges.")
        return edge_dict_list, nodes

    def extract_with_srtk(
        self, 
        seed_entities: List[str],
        question: str,
        max_hops,
        beam_size,
        max_nodes,
        hypothetical_answer: str,
        compare_to_hypothetical_answer: bool = False,
    ):
        if compare_to_hypothetical_answer:
            print("Hypothetical Answer:", hypothetical_answer)
            q_emb = self.similarity_model.encode(hypothetical_answer)
        else:
            q_emb = self.similarity_model.encode(question, show_progress_bar=False)

        triples = []
        seeds = [entity for entity in seed_entities if self.graph.has_node(entity)]
        seen_edges = set()
        seen_nodes = set(seeds)
        frontier = set(seeds)

        curr_beam_size = beam_size

        for hop in range(max_hops):
            candidates = []

            for node in frontier:
                neighbors = list(nx.bfs_edges(self.graph, node, depth_limit=1))
                for pair in neighbors:
                    for i in range(self.graph.number_of_edges(pair[0], pair[1])):
                        edge_dict = {
                            "from": pair[0],
                            "to": pair[1],
                            "label": self.graph.edges[pair[0], pair[1], i]["label"],
                            "description": self.graph.edges[pair[0], pair[1], i]["description"],
                        }
                        if self.kg_name in ["wikidata", "umls"]:                                                # directed graphs
                            key = (edge_dict["from"], edge_dict["label"], edge_dict["to"])
                        else:                                                                                   # undirected graphs
                            key = tuple(sorted([edge_dict["from"], edge_dict["to"]])) + (edge_dict["label"],)
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)

                        score = path_similarity(q_emb, edge_dict["description"], self.similarity_model)
                        candidates.append((score, edge_dict))

            if not candidates:
                break

            # sort by similarity and keep top beam_size
            candidates.sort(key=lambda x: x[0], reverse=True)
            keep = candidates[:curr_beam_size]

            new_frontier = set()
            for score, edge in keep:
                triples.append(edge)
                new_frontier.add(edge["to"])
                seen_nodes.add(edge["from"])
                seen_nodes.add(edge["to"])
                if len(triples) >= max_nodes:
                    break

            frontier = new_frontier
            if len(triples) >= max_nodes or not frontier:
                break

            # curr_beam_size = curr_beam_size * beam_size

        return triples, seen_nodes

    def extract_with_srtk_cumulative_context(
        self, seed_entities: List[str], question, max_hops, beam_size, max_nodes, hypothetical_answer: str, compare_to_hypothetical_answer: bool = False,
    ):
        if compare_to_hypothetical_answer:
            print("Hypothetical Answer:", hypothetical_answer)
            q_emb = self.similarity_model.encode(hypothetical_answer)
        else:
            q_emb = self.similarity_model.encode(question, show_progress_bar=False)

        triples = []
        seeds = [entity for entity in seed_entities if self.graph.has_node(entity)]
        seen_edges = set()
        seen_nodes = set(seeds)

        frontier = {(seed, "") for seed in seeds}

        curr_beam_size = beam_size

        # print(f"\n[INFO] Starting cumulative-context SRTK retrieval with {len(seeds)} seeds")
        # print(f"[INFO] Question: {question}\n")

        for hop in range(max_hops):
            # print(f"\n=== HOP {hop+1}/{max_hops} ===")
            candidates = []

            for node, cum_desc in frontier:
                neighbors = list(nx.bfs_edges(self.graph, node, depth_limit=1))
                for pair in neighbors:
                    for i in range(self.graph.number_of_edges(pair[0], pair[1])):
                        edge_dict = {
                            "from": pair[0],
                            "to": pair[1],
                            "label": self.graph.edges[pair[0], pair[1], i]["label"],
                            "description": self.graph.edges[pair[0], pair[1], i]["description"],
                        }

                        # Unique key: directed or undirected
                        if self.kg_name in ["wikidata", "umls"]:
                            key = (edge_dict["from"], edge_dict["label"], edge_dict["to"])
                        else:
                            key = tuple(sorted([edge_dict["from"], edge_dict["to"]])) + (edge_dict["label"],)
                        if key in seen_edges:
                            continue
                        seen_edges.add(key)

                        # Build cumulative description
                        new_desc = f"{cum_desc}; {edge_dict['description']}"
                        score = path_similarity(q_emb, new_desc, self.similarity_model)
                        candidates.append((score, edge_dict, new_desc))

            if not candidates:
                print("[INFO] No more candidates to expand.")
                break

            # sort by similarity and keep top beam_size
            candidates.sort(key=lambda x: x[0], reverse=True)
            keep = candidates[:curr_beam_size]

            # print(f"[INFO] {len(candidates)} candidates → keeping top {len(keep)} edges.")
            # print("[DEBUG] Top 3 edges:")
            # for score, edge, description in keep[:3]:
            #     print(f"  ({edge['from']} -[{edge['label']}]-> {edge['to']}) \n    Description: {description} \n    Score={score:.4f}")

            new_frontier = set()
            for score, edge, new_desc in keep:
                triples.append(edge)
                new_frontier.add((edge["to"], new_desc))
                seen_nodes.add(edge["from"])
                seen_nodes.add(edge["to"])
                if len(triples) >= max_nodes:
                    break

            frontier = new_frontier
            if len(triples) >= max_nodes or not frontier:
                print(f"Triples collected: {len(triples)}")
                print("[INFO] Reached node/edge limit or no frontier left.")
                break

            # curr_beam_size = curr_beam_size * beam_size

        print(f"\n[INFO] Completed retrieval. Collected {len(triples)} edges, {len(seen_nodes)} nodes.")
        return triples, seen_nodes



def path_similarity(question_embedding, context, similarity_model):
    context_embedding = similarity_model.encode(context, show_progress_bar=False)
    return cosine_similarity(np.array([question_embedding], dtype=object), np.array([context_embedding], dtype=object))[0][0]


#########################################################################
############# Evaluation and Ground-Truth Loading Functions #############
#########################################################################

def evaluate_subgraph_extraction(
    gt_file: str,
    pred_edges: List[Dict],
    undirected: bool = True
) -> Tuple[float, float, float]:
    """
    Evaluate precision, recall, and F1 for edge-based subgraph extraction.
    """

    gt_edges = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # <-- skip comments/seeds
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            head, relation, tail = parts
            gt_edges.append({"from": head, "label": relation, "to": tail})

    # Normalize function
    def normalize(edge: Dict) -> Tuple[str, str, str]:
        if undirected:
            nodes = sorted([edge["from"], edge["to"]])
            return (nodes[0], edge["label"], nodes[1])
        else:
            return (edge["from"], edge["label"], edge["to"])

    gt_set = {normalize(e) for e in gt_edges}
    pred_set = {normalize(e) for e in pred_edges}

    # Calculate overlaps
    true_pos = len(gt_set & pred_set)
    print("True Positives:", true_pos)
    print("\nSubgraph Edges:", len(pred_set))
    precision = true_pos / len(pred_set) if pred_set else 0.0
    recall = true_pos / len(gt_set) if gt_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    missing = gt_set - pred_set
    if missing:
        print("\n[False Negatives: edges in ground truth but NOT retrieved]")
        for edge in missing:
            print(f"  {edge[0]} | {edge[1]} | {edge[2]}")
    else:
        print("\nAll ground-truth edges were covered (no false negatives).")

    return precision, recall, f1


#############################################################################

def get_nodes_and_edges_matching_gt(gt_file: str, pred_edges: List[Dict], directed: bool):
    """
    Return the list of ground-truth edges that are present in the predicted edges (true positives).
    """

    gt_edges = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):   # <-- skip comments/seeds
                continue
            parts = line.split("|")
            if len(parts) != 3:
                continue
            head, relation, tail = parts
            gt_edges.append({"from": head, "label": relation, "to": tail})

    # Normalize function
    def normalize(edge: Dict) -> Tuple[str, str, str]:
        if directed:
            return (edge["from"], edge["label"], edge["to"])
        else:
            nodes = sorted([edge["from"], edge["to"]], key=lambda x: x.lower())
            return (nodes[0], edge["label"], nodes[1])

    gt_set = {normalize(e) for e in gt_edges}
    pred_set = {normalize(e) for e in pred_edges}

    # true_positives = list(gt_set & pred_set)
    true_positives = [f"{edge[0]} {edge[1]} {edge[2]}" for edge in (gt_set & pred_set)]

    return [], true_positives

#############################################################################

def get_experiment_llm_answers(answers_file: str) -> Tuple[List[str], List[str]]:
    """
    Parse an experiment file containing LLM answers and reasoning steps.
    """
    answers: List[str] = []
    reasoning_steps: List[str] = []

    with open(answers_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines

            if line.startswith("#ANSWERS:"):
                # Parse answers from the first line
                ans_part = line[len("#ANSWERS:"):].strip()
                answers = [a.strip() for a in ans_part.split("|") if a.strip()]
            else:
                # Every other line is a reasoning step
                reasoning_steps.append(line)

    return answers, reasoning_steps

#############################################################################

def load_ground_truth_subgraph(gt_file: str) -> Tuple[List[Dict], Set[str], List[str], List[str]]:
    """
    Load the ground-truth subgraph from a text file of triples.
    First line may contain seeds as: #SEEDS: node1|node2|...
    """
    edge_dict_list: List[Dict] = []
    nodes_set: Set[str] = set()
    edge_descriptions: List[str] = []
    seed_nodes: List[str] = []

    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#SEEDS:"):
                # parse seeds
                seeds = line[len("#SEEDS:"):].strip()
                seed_nodes = [s.strip() for s in seeds.split("|") if s.strip()]
                continue

            parts = line.split("|")
            if len(parts) != 3:
                continue
            head, relation, tail = parts

            edge = {
                "from": head,
                "to": tail,
                "label": relation,
                "description": create_meta_qa_description(head, relation, tail)
            }
            edge_dict_list.append(edge)
            nodes_set.update([head, tail])
            edge_descriptions.append(edge["description"])

    return seed_nodes, nodes_set, edge_dict_list, edge_descriptions, 0


# This is a copy of the create_description function in metaqa.py to create edge descriptions for MetaQA dataset in experiments. 
def create_meta_qa_description(head, relation, tail):
    if relation == "directed_by":
        return "Movie \"{}\" was directed by \"{}\"".format(head, tail)
    elif relation == "has_genre":
        return "Movie \"{}\" has genre {}".format(head, tail)
    elif relation == "has_imdb_rating":
        return "Movie \"{}\" is rated {} in imdb".format(head, tail)
    elif relation == "has_imdb_votes":
        return "Movie \"{}\" is voted {} in imdb".format(head, tail)
    elif relation == "has_tags":
        return "Movie \"{}\" is tagged with \"{}\"".format(head, tail)
    elif relation == "in_language":
        return "Movie \"{}\" is in {} language".format(head, tail)
    elif relation == "release_year":
        return "Movie \"{}\" was released in {}".format(head, tail)
    elif relation == "starred_actors":
        return "Actor \"{}\" starred in \"{}\"".format(tail, head)
    elif relation == "written_by":
        return "Movie \"{}\" was written by \"{}\"".format(head, tail)
    else:
        return ""