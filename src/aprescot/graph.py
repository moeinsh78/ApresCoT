import json
from typing import Tuple, Dict, List
import networkx as nx
import pandas as pd
import numpy as np
import requests
from queue import PriorityQueue

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


import time
import requests

WDQS = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "ApresCoT-Subgraph/1.0 (mailto:you@example.com)"
}


BALANCED = {
    "P31","P279","P361","P527","P921",
    "P131","P17","P495","P276",
    "P170","P57","P58","P86","P50","P161","P1029",
    "P136","P2283",
    "P571","P577",
    "P463","P102","P108","P39","P106",
    "P530","P1376","P155","P156","P641"
}


class WikiDataKnowledgeGraph:
    def run_sparql(self, query, retry=3, backoff=0.4):
        last_err = None
        for i in range(retry):
            try:
                r = requests.get(WDQS, params={"query": query}, headers=HEADERS, timeout=60)
                if r.status_code == 200:
                    return r.json()["results"]["bindings"]
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                last_err = e
            time.sleep(backoff * (i + 1))
        raise last_err

    def _qid(self, uri: str) -> str:
        return uri.rsplit("/", 1)[-1]

    def _pid(self, uri: str) -> str:
        return uri.rsplit("/", 1)[-1]

    def one_hop(self, seed_q, limit=200, p_whitelist=None):
        # Outgoing: seed wdt:P ?o (entity)
        q_out = f"""
        SELECT ?p ?o WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?seed ?p ?o .
        FILTER(STRSTARTS(STR(?p), STR(wdt:)))
        FILTER(STRSTARTS(STR(?o), STR(wd:)))
        }} LIMIT {limit*2}
        """
        # Incoming: ?s wdt:P seed (entity)
        q_in = f"""
        SELECT ?s ?p WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?s ?p ?seed .
        FILTER(STRSTARTS(STR(?p), STR(wdt:)))
        FILTER(STRSTARTS(STR(?s), STR(wd:)))
        }} LIMIT {limit*2}
        """

        out = self.run_sparql(q_out)
        inn = self.run_sparql(q_in)

        out_edges, in_edges = [], []
        for b in out:
            p = self._pid(b["p"]["value"])
            o = self._qid(b["o"]["value"])
            if (p_whitelist is None) or (p in p_whitelist):
                out_edges.append((seed_q, p, o))
                if len(out_edges) >= limit: break

        for b in inn:
            s = self._qid(b["s"]["value"])
            p = self._pid(b["p"]["value"])
            if (p_whitelist is None) or (p in p_whitelist):
                in_edges.append((s, p, seed_q))
                if len(in_edges) >= limit: break

        return out_edges, in_edges

    def two_hop(self, seed_q, per_path_limit=400, p_whitelist=None):
        # Outgoing→Outgoing: seed -p1-> m -p2-> x
        q_oo = f"""
        SELECT ?p1 ?m ?p2 ?x WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?seed ?p1 ?m .
        ?m ?p2 ?x .
        FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
        FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
        FILTER(STRSTARTS(STR(?m), STR(wd:)))
        FILTER(STRSTARTS(STR(?x), STR(wd:)))
        }} LIMIT {per_path_limit*2}
        """
        # Incoming→Incoming: s -p1-> seed ; y -p2-> s
        q_ii = f"""
        SELECT ?y ?p2 ?s ?p1 WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?s ?p1 ?seed .
        ?y ?p2 ?s .
        FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
        FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
        FILTER(STRSTARTS(STR(?s), STR(wd:)))
        FILTER(STRSTARTS(STR(?y), STR(wd:)))
        }} LIMIT {per_path_limit*2}
        """
        # Outgoing→Incoming: seed -p1-> m ; y -p2-> m
        q_oi = f"""
        SELECT ?p1 ?m ?y ?p2 WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?seed ?p1 ?m .
        ?y ?p2 ?m .
        FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
        FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
        FILTER(STRSTARTS(STR(?m), STR(wd:)))
        FILTER(STRSTARTS(STR(?y), STR(wd:)))
        }} LIMIT {per_path_limit*2}
        """
        # Incoming→Outgoing: s -p1-> seed ; s -p2-> x
        q_io = f"""
        SELECT ?s ?p1 ?p2 ?x WHERE {{
        VALUES ?seed {{ wd:{seed_q} }}
        ?s ?p1 ?seed .
        ?s ?p2 ?x .
        FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
        FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
        FILTER(STRSTARTS(STR(?s), STR(wd:)))
        FILTER(STRSTARTS(STR(?x), STR(wd:)))
        }} LIMIT {per_path_limit*2}
        """

        results = {
            "oo": self.run_sparql(q_oo),
            "ii": self.run_sparql(q_ii),
            "oi": self.run_sparql(q_oi),
            "io": self.run_sparql(q_io),
        }

        edges = []
        def keep_p(p): return (p_whitelist is None) or (p in p_whitelist)

        for b in results["oo"]:
            p1 = self._pid(b["p1"]["value"]); m = self._qid(b["m"]["value"])
            p2 = self._pid(b["p2"]["value"]); x = self._qid(b["x"]["value"])
            if keep_p(p1) and keep_p(p2):
                edges.append((seed_q, p1, m))
                edges.append((m, p2, x))
        for b in results["ii"]:
            y = self._qid(b["y"]["value"]); p2 = self._pid(b["p2"]["value"])
            s = self._qid(b["s"]["value"]); p1 = self._pid(b["p1"]["value"])
            if keep_p(p1) and keep_p(p2):
                edges.append((s, p1, seed_q))
                edges.append((y, p2, s))
        for b in results["oi"]:
            p1 = self._pid(b["p1"]["value"]); m = self._qid(b["m"]["value"])
            y  = self._qid(b["y"]["value"]);  p2 = self._pid(b["p2"]["value"])
            if keep_p(p1) and keep_p(p2):
                edges.append((seed_q, p1, m))
                edges.append((y, p2, m))
        for b in results["io"]:
            s = self._qid(b["s"]["value"]); p1 = self._pid(b["p1"]["value"])
            p2 = self._pid(b["p2"]["value"]); x = self._qid(b["x"]["value"])
            if keep_p(p1) and keep_p(p2):
                edges.append((s, p1, seed_q))
                edges.append((s, p2, x))

        # Dedup
        edges = list(dict.fromkeys(edges))
        return edges

    def fetch_labels(self, qids, pids, lang="en"):
        labels = {}
        qids = list(set(qids))
        pids = list(set(pids))

        # Entity labels
        for i in range(0, len(qids), 200):
            vals = " ".join(f"wd:{q}" for q in qids[i:i+200])
            q = f"""
            SELECT ?e ?eLabel WHERE {{
            VALUES ?e {{ {vals} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
            }}
            """
            for b in self.run_sparql(q):
                labels[self._qid(b["e"]["value"])] = b.get("eLabel", {}).get("value", "")

        # Property labels
        for i in range(0, len(pids), 200):
            vals = " ".join(f"wd:{p}" for p in pids[i:i+200])
            q = f"""
            SELECT ?p ?pLabel WHERE {{
            VALUES ?p {{ {vals} }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
            }}
            """
            for b in self.run_sparql(q):
                labels[self._pid(b["p"]["value"])] = b.get("pLabel", {}).get("value", "")
            
        return labels

    def extract_subgraph_two_hops(
        self,
        seed_qids,
        one_hop_limit=200,
        two_hop_limit=400,
        p_whitelist=None,
        label_lang="en",
        total_edge_cap=8000
    ):
        nodes = set(seed_qids)
        edges = []

        # 1-hop (both directions)
        for q in seed_qids:
            out_e, in_e = self.one_hop(q, limit=one_hop_limit, p_whitelist=p_whitelist)
            edges.extend(out_e); edges.extend(in_e)
            for s,p,o in out_e: nodes.update([s,o])
            for s,p,o in in_e:  nodes.update([s,o])

            if len(edges) >= total_edge_cap: break

        # 2-hop (four patterns) rooted at each seed
        if len(edges) < total_edge_cap:
            for q in seed_qids:
                e2 = self.two_hop(q, per_path_limit=two_hop_limit, p_whitelist=p_whitelist)
                edges.extend(e2)
                for s,p,o in e2: nodes.update([s,o])
                if len(edges) >= total_edge_cap: break

        # Dedup & trim
        edges = list(dict.fromkeys(edges))[:total_edge_cap]

        # Gather labels
        qids = {n for n in nodes if n.startswith("Q")}
        pids = {p for _,p,_ in edges}
        labels = self.fetch_labels(list(qids), list(pids), lang=label_lang)

        edges = [(s, labels.get(p, p), o) for (s, p, o) in edges]

        return {
            "nodes": list(nodes),
            "edges": edges,          # (subjectQ, predicateP, objectQ)
            "labels": labels,        # Q/P -> label
            "meta": {
                "seeds": seed_qids,
                "one_hop_limit": one_hop_limit,
                "two_hop_limit": two_hop_limit,
                "total_edge_cap": total_edge_cap,
                "predicate_whitelist": sorted(list(p_whitelist)) if p_whitelist else None
            }
        }



    def search_wikidata_entity(self, query, language="en"):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "search": query
        }
        r = requests.get(url, params=params)
        results = r.json().get("search", [])

        if results:
            return results[0]["id"], results[0]["label"], results[0].get("description", "")

        return None

    def find_wikidata_entities(self, named_entities: List[str]) -> List[str]:
        entities_ids = []
        for entity in named_entities:
            entity_id = self.search_wikidata_entity(entity)
            if entity_id:
                entities_ids.append(entity_id)
            else:
                print(f"Entity '{entity}' not found in Wikidata.")

        return entities_ids
    
    def extract_relevant_subgraph(self, seed_qids: List[str]) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]], List[str]]:
        # Optional: keep it general or pass a domain-specific prop whitelist
        p_whitelist = BALANCED
        sg = self.extract_subgraph_two_hops(
            seed_qids,
            one_hop_limit=80,
            two_hop_limit=150,
            p_whitelist=p_whitelist,
            total_edge_cap=2000
        )
        print(f"Nodes: {len(sg['nodes'])}, Edges: {len(sg['edges'])}")
        L = sg["labels"]
        edges_labeled = [
            (L.get(s, s), L.get(p, p), L.get(o, o))
            for (s, p, o) in sg["edges"]
        ]
        for s, p, o in edges_labeled[:200]:
            print(f"({s}) -[{p}]-> ({o})")


class UMLSKnowledgeGraph:
    kg_directory = "kg/umls-kb.txt"


    def create_description(self, head, relation, tail):
        relation_label = relation.replace("_", " ")
        if relation_label == "isa": 
            return "{} is a {}.".format(head, tail)
        else:
            return "{} {} {}.".format(head, relation_label, tail)


    def load_kg_edges_df(self, edge_list_file):
        kg_relations = pd.read_table(edge_list_file, delimiter = "|", header=None)
        kg_relations = kg_relations.rename(columns = {0:"head", 1:"relation", 2:"tail"})
        kg_relations['description'] = kg_relations.apply(lambda d: self.create_description(d["head"], d["relation"], d["tail"]), axis = 1)
        
        return kg_relations
    
    
    def get_bfs_subgraph(self, graph: nx.MultiDiGraph, source_node: str, depth: int) -> list[dict]:
        print("Search begins starting from Node: ", source_node)
        # ending_node_relations = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]
        edge_dict_list = []
        to_be_expanded = [source_node]
        visited = set()
        curr_depth = 0
        nodes = set()
        nodes.add(source_node)
        while curr_depth < depth:
            to_expand_count = len(to_be_expanded)
            for _ in range(to_expand_count):
                curr_node = to_be_expanded.pop(0)
                if curr_node in visited:
                    continue
                neighbors = list(graph.out_edges(curr_node))
                visited.add(curr_node)
                for pair in neighbors:
                    # Node to be expanded is always in the second position
                    if pair[1] in visited:
                        continue
                    edge = {}
                    to_be_expanded.append(pair[1])
                    for i in range(graph.number_of_edges(pair[0], pair[1])):
                        nodes.add(pair[1])
                        edge = {}
                        edge["from"] = pair[0]
                        edge["to"] = pair[1]
                        edge["label"] = graph.edges[pair[0], pair[1], i]["label"]
                        edge["description"] = graph.edges[pair[0], pair[1], i]["description"]
                        edge_dict_list.append(edge)
            
            curr_depth += 1
            
        return edge_dict_list
    
    
    
    def build_knowledge_graph(self, edge_list_file):
        # Graph Construction
        graph = nx.MultiDiGraph()
        edges = self.load_kg_edges_df(edge_list_file)
        for _, edge in edges.iterrows():
            graph.add_edge(edge['head'], edge['tail'], label=edge['relation'], description=edge["description"])
    
        return graph
    
    
    def extract_surrounding_subgraph(self, seed_entities, depth = 2):
        graph = self.build_knowledge_graph(edge_list_file = self.kg_directory)
    
        edge_dict_list = []
        for entity in seed_entities:
            edges = self.get_bfs_subgraph(graph, entity, depth)
            edge_dict_list.extend(edges)
    
        return edge_dict_list, self.get_nodes_set(edge_dict_list)
    


    def extract_subgraph_edge_descriptions(self, edge_dict_list):
        edge_desc_list = []

        for edge_dict in edge_dict_list:    
            edge_desc_list.append(edge_dict["description"])

        return edge_desc_list



    def get_similarity_based_subgraph(
            self, graph, similarity_model, question, seed_entities, depth, 
            not_to_expand_relation_labels = [], path_similarity_cutoff = 0.2
        ):

        edge_list = []
        question_embedding = similarity_model.encode(question, show_progress_bar=False)
        not_to_expand_relation_set = set(not_to_expand_relation_labels)

        # Max heap for candidate edges
        path_pool = PriorityQueue() 
        visited = set()

        for entity in seed_entities:
            if not graph.has_node(entity):
                continue
            visited.add(entity)
            neighbors = list(graph.out_edges(entity))
            for pair in neighbors:
                for i in range(graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = graph.edges[pair[0], pair[1], i]["description"]
                    
                    path_description = edge_dict["description"]
                    path_pool.put((-path_similarity(question_embedding, path_description, similarity_model), edge_dict, path_description))

        print_pool(path_pool)

        while not path_pool.empty():
            most_similar_path = path_pool.get()

            print("Most Similar Path: ", most_similar_path)
            curr_similarity_score = (-1) * most_similar_path[0]
            most_relevant_edge = most_similar_path[1]
            most_relevant_edge_description = most_similar_path[2]
            # if ((most_similar_path[0]) * (-1) < path_similarity_cutoff):
            #     break
            
            if (len(edge_list) >= 100):
                break

            if (most_relevant_edge["to"] in visited):
                continue

            # Adding the most relevant edge to the subgraph
            edge_list.append(most_similar_path[1])
            print("\nAdding Edge to Subgraph: ", most_similar_path[1])

            if most_similar_path[1]["label"] in not_to_expand_relation_set:
                continue

            # Adding new node's neighbours to possible paths
            node_to_be_expanded = most_relevant_edge["to"]
            visited.add(node_to_be_expanded)
            neighbors = list(nx.bfs_edges(graph, node_to_be_expanded, depth_limit=1))
            for pair in neighbors:
                for i in range(graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = graph.edges[pair[0], pair[1], i]["description"]
                    
                    path_description = most_relevant_edge_description + " " + edge_dict["description"]
                    new_path_similarity = path_similarity(question_embedding, path_description, similarity_model)
                    print(f"New Path Similarity ({pair[0]}, {pair[1]}, {edge_dict['label']}): ", new_path_similarity)
                    path_pool.put(((-1) * curr_similarity_score * new_path_similarity, edge_dict, path_description))

        
        print("Final Edge List:\n")
        for edge in edge_list:
            print(edge)
        return edge_list


    def get_nodes_set(self, edge_dict_list: List[Dict]):
        nodes_set = set()
        for edge in edge_dict_list:
            print(edge)
            print("-")
            nodes_set.add(edge["from"])
            nodes_set.add(edge["to"])
        
        return nodes_set


    def extract_relevant_subgraph(self, seed_entities, question, depth = 2):
        graph = self.build_knowledge_graph(edge_list_file = self.kg_directory)
        similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        edge_dict_list = self.get_similarity_based_subgraph(graph, similarity_model, question, seed_entities, depth, 
                                                            not_to_expand_relation_labels = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"])
        
        return edge_dict_list, self.get_nodes_set(edge_dict_list)
                    


class MetaQAKnowledgeGraph:
    kg_directory = "kg/meta-qa-kb.txt"

    def create_description(self, head, relation, tail):
        if relation == "directed_by":
            return "Movie \"{}\" was directed by \"{}\".".format(head, tail)
        elif relation == "has_genre":
            return "Movie \"{}\" has genre {}.".format(head, tail)
        elif relation == "has_imdb_rating":
            return "Movie \"{}\" is rated {} in imdb.".format(head, tail)
        elif relation == "has_imdb_votes":
            return "Movie \"{}\" is voted {} in imdb.".format(head, tail)
        elif relation == "has_tags":
            return "Movie \"{}\" is described with \"{}\" tag.".format(head, tail)
        elif relation == "in_language":
            return "Movie \"{}\" is in {} language.".format(head, tail)
        elif relation == "release_year":
            return "Movie \"{}\" was released in {}.".format(head, tail)
        elif relation == "starred_actors":
            return "Actor \"{}\" starred in \"{}\".".format(tail, head)
        elif relation == "written_by":
            return "Movie \"{}\" was written by \"{}\".".format(head, tail)
        else:
            print("ERROR!: Relation type {} unknown!".format(relation))


    def load_kg_edges_df(self, edge_list_file):
        kg_relations = pd.read_table(edge_list_file, delimiter = "|", header=None)
        kg_relations = kg_relations.rename(columns = {0:"head", 1:"relation", 2:"tail"})
        kg_relations['description'] = kg_relations.apply(lambda d: self.create_description(d["head"], d["relation"], d["tail"]), axis = 1)
        
        return kg_relations
    
    
    def get_bfs_subgraph(self, graph: nx.MultiGraph, source_node: str, depth: int, expand_ending_nodes: bool = False) -> list[dict]:
        print("Search begins from Node: ", source_node)
        ending_node_relations = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"]
        edge_dict_list = []
        to_be_expanded = [source_node]
        visited = set()
        curr_depth = 0
        nodes = set()
        nodes.add(source_node)
        while curr_depth < depth:
            to_expand_count = len(to_be_expanded)
            for _ in range(to_expand_count):
                curr_node = to_be_expanded.pop(0)
                if curr_node in visited:
                    continue
                neighbors = list(nx.bfs_edges(graph, curr_node, depth_limit=1))
                visited.add(curr_node)
                for pair in neighbors:
                    # Node to be expanded is always in the second position
                    if pair[1] in visited:
                        continue
                    edge = {}
                    if (expand_ending_nodes) or (graph.edges[pair[0], pair[1], 0]["label"] not in ending_node_relations):
                        to_be_expanded.append(pair[1])
                    for i in range(graph.number_of_edges(pair[0], pair[1])):
                        nodes.add(pair[1])
                        edge = {}
                        edge["from"] = pair[0]
                        edge["to"] = pair[1]
                        edge["label"] = graph.edges[pair[0], pair[1], i]["label"]
                        edge["description"] = graph.edges[pair[0], pair[1], i]["description"]
                        edge_dict_list.append(edge)
            
            curr_depth += 1
            
        return edge_dict_list
    
    
    
    def build_knowledge_graph(self, edge_list_file):
        # Graph Construction
        graph = nx.MultiGraph()
        edges = self.load_kg_edges_df(edge_list_file)
        for _, edge in edges.iterrows():
            graph.add_edge(edge['head'], edge['tail'], label=edge['relation'], description=edge["description"])
    
        return graph
    
    
    def extract_surrounding_subgraph(self, seed_entities, depth = 2):
        graph = self.build_knowledge_graph(edge_list_file = self.kg_directory)
    
        edge_dict_list = []
        for entity in seed_entities:
            edges = self.get_bfs_subgraph(graph, entity, depth, expand_ending_nodes = False)
            edge_dict_list.extend(edges)
    
        return edge_dict_list, self.get_nodes_set(edge_dict_list)
    


    def extract_subgraph_edge_descriptions(self, edge_dict_list):
        edge_desc_list = []

        for edge_dict in edge_dict_list:    
            edge_desc_list.append(edge_dict["description"])

        return edge_desc_list



    def get_similarity_based_subgraph(self, graph, similarity_model, question, seed_entities, 
                                      depth, not_to_expand_relation_labels = [], path_similarity_cutoff = 0.2):
        edge_list = []
        question_embedding = similarity_model.encode(question, show_progress_bar=False)
        not_to_expand_relation_set = set(not_to_expand_relation_labels)

        # Max heap for candidate edges
        path_pool = PriorityQueue() 
        visited = set()

        for entity in seed_entities:
            visited.add(entity)
            neighbors = list(nx.bfs_edges(graph, entity, depth_limit=1))
            for pair in neighbors:
                for i in range(graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = graph.edges[pair[0], pair[1], i]["description"]
                    
                    path_description = edge_dict["description"]
                    path_pool.put((-path_similarity(question_embedding, path_description, similarity_model), edge_dict, path_description))

        print_pool(path_pool)

        while not path_pool.empty():
            most_similar_path = path_pool.get()

            print("Most Similar Path: ", most_similar_path)
            curr_similarity_score = (-1) * most_similar_path[0]
            most_relevant_edge = most_similar_path[1]
            most_relevant_edge_description = most_similar_path[2]
            # if ((most_similar_path[0]) * (-1) < path_similarity_cutoff):
            #     break
            
            if (len(edge_list) >= 30):
                break

            if (most_relevant_edge["to"] in visited):
                continue

            # Adding the most relevant edge to the subgraph
            edge_list.append(most_similar_path[1])
            print("\nAdding Edge to Subgraph: ", most_similar_path[1])

            if most_similar_path[1]["label"] in not_to_expand_relation_set:
                continue

            # Adding new node's neighbours to possible paths
            node_to_be_expanded = most_relevant_edge["to"]
            visited.add(node_to_be_expanded)
            neighbors = list(nx.bfs_edges(graph, node_to_be_expanded, depth_limit=1))
            for pair in neighbors:
                for i in range(graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = graph.edges[pair[0], pair[1], i]["description"]
                    
                    path_description = most_relevant_edge_description + " " + edge_dict["description"]
                    new_path_similarity = path_similarity(question_embedding, path_description, similarity_model)
                    print(f"New Path Similarity ({pair[0]}, {pair[1]}, {edge_dict['label']}): ", new_path_similarity)
                    path_pool.put(((-1) * curr_similarity_score * new_path_similarity, edge_dict, path_description))
                    # if (len(edge_list) >= 30):
                    #     break

        
        print("Final Edge List:\n")
        for edge in edge_list:
            print(edge)
        return edge_list


    def get_nodes_set(self, edge_dict_list: List[Dict]):
        nodes_set = set()
        for edge in edge_dict_list:
            # print(edge)
            # print("-")
            nodes_set.add(edge["from"])
            nodes_set.add(edge["to"])
        
        return nodes_set


    def extract_relevant_subgraph(self, seed_entities, question, depth = 2):
        graph = self.build_knowledge_graph(edge_list_file = self.kg_directory)
        similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        edge_dict_list = self.get_similarity_based_subgraph(graph, similarity_model, question, seed_entities, depth, 
                                                            not_to_expand_relation_labels = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"])
        
        return edge_dict_list, self.get_nodes_set(edge_dict_list)
    


def print_pool(path_pool: PriorityQueue[Tuple]):
    print("#############################################\nPrinting the whole pool queue...")
    
    for i in range(path_pool.qsize()):
        
        print("Similarity Score:", path_pool.queue[i][0])
        print("Edge Dictionary:", path_pool.queue[i][1])
        print("Path Description:", path_pool.queue[i][2])
        print("\n")

    print("Printing Complete...\n#############################################")


def path_similarity(question_embedding, context, similarity_model):
    context_embedding = similarity_model.encode(context, show_progress_bar=False)
    return cosine_similarity(np.array([question_embedding], dtype=object), np.array([context_embedding], dtype=object))[0][0]