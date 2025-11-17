from typing import List, Dict, Tuple, Set
import numpy as np
import networkx as nx
from openai import OpenAI

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from src.aprescot.metaqa import create_description as create_metaqa_description


KG_DIRECTORY = {
    "meta-qa": "kg/meta-qa-kb.txt",
    "wikidata": "kg/wikidata-cached.txt",
    "umls": "kg/umls-kg.txt",
}

class DemoSubgraphRetriever:
    def __init__(self, kg_name: str, scorer_model: str, model_cache_folder: str):
        self.kg_name = kg_name
        self.graph = self.load_graph_from_file(edge_list_file=KG_DIRECTORY[kg_name])
        self.similarity_model = SentenceTransformer(
            scorer_model, cache_folder=model_cache_folder
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
                    description = create_metaqa_description(head, relation, tail)
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
