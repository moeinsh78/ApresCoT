from typing import Tuple
import networkx as nx
import pandas as pd
import numpy as np
import requests
from queue import PriorityQueue

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class UMLSKnowledgeGraph:
    def __init__(self, apikey="f1647e11-1f12-4997-83d7-04b15a1d14f2", version="current"):
        self.apikey = apikey
        self.version = version
        self.search_url = f"https://uts-ws.nlm.nih.gov/rest/search/{version}"
        self.content_url = f"https://uts-ws.nlm.nih.gov/rest/content/{version}"
        self.content_suffix = "/CUI/{}/{}?apiKey={}"

    def search_cui(self, query):
        cui_results = []

        try:
            page = 1
            size = 1
            query = {"string": query, "apiKey": self.apikey, "pageNumber": page, "pageSize": size}
            r = requests.get(self.search_url, params=query)
            r.raise_for_status()
            print(r.url)
            r.encoding = 'utf-8'
            outputs = r.json()

            items = outputs["result"]["results"]

            if len(items) == 0:
                print("No results found.\n")

            for result in items:
                cui_results.append((result["ui"], result["name"]))

        except Exception as except_error:
            print("We got an error: ", except_error)
            print(except_error)

        return cui_results

    def get_definitions(self, cui):
        try:
            suffix = self.content_suffix.format(cui, "definitions", self.apikey)
            r = requests.get(self.content_url + suffix)
            r.raise_for_status()
            r.encoding = "utf-8"
            outputs = r.json()

            return outputs["result"]
        except Exception as except_error:
            print(except_error)


    def get_relations(self, cui, pages=3):
        all_relations = []

        try:
            for page in range(1, pages + 1):
                suffix = self.content_suffix.format(cui, "relations", self.apikey) + f"&pageNumber={page}"
                r = requests.get(self.content_url + suffix)
                r.raise_for_status()
                r.encoding = "utf-8"
                outputs = r.json()

                page_relations = outputs.get("result", [])
                all_relations.extend(page_relations)

        except Exception as except_error:
            print(except_error)

        return all_relations


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
        print("Search begins starting from Node: ", source_node)
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
            
        return edge_dict_list, nodes
    
    
    
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
            edges, nodes_set = self.get_bfs_subgraph(graph, entity, depth, expand_ending_nodes = False)
            edge_dict_list.extend(edges)
    
        return edge_dict_list, nodes_set
    


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

        
        # print_pool(path_pool)
        print("Final Edge List:\n")
        for edge in edge_list:
            print(edge)
        return edge_list, self.get_nodes_set(edge_list)


    def get_nodes_set(self, edge_list):
        nodes_set = set()
        for edge in edge_list:
            nodes_set.add(edge["from"])
            nodes_set.add(edge["to"])
        
        return nodes_set


    def extract_relevant_subgraph(self, seed_entities, question, depth = 2):
        graph = self.build_knowledge_graph(edge_list_file = self.kg_directory)
        similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        edge_dict_list, nodes_set = self.get_similarity_based_subgraph(graph, similarity_model, question, seed_entities, depth, 
                                                            not_to_expand_relation_labels = ["release_year", "in_language", "has_tags", "has_genre", "has_imdb_rating", "has_imdb_votes"])
        
        # for entity in seed_entities:
        #     edges, nodes_set = self.get_bfs_subgraph(graph, entity, depth, expand_ending_nodes = True)
        #     edge_dict_list.extend(edges)

    
        return edge_dict_list, nodes_set
    


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