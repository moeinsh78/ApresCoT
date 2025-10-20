from typing import Tuple, Dict, List
import networkx as nx
import pandas as pd
import numpy as np
from queue import PriorityQueue

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from torch import topk
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
HF_MODELS_DIR = os.path.join(PROJECT_ROOT, 'hf_models')

class UMLSKnowledgeGraph:
    def __init__(self, kg_directory: str = "kg/umls-kb.txt"):
        self.graph = self.build_knowledge_graph(edge_list_file = kg_directory)

    def node_count(self):
        return self.graph.number_of_nodes()
    
    def edge_count(self):
        return self.graph.number_of_edges()
    
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
    
    def get_bfs_subgraph(self, source_node: str, depth: int) -> list[dict]:
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
                neighbors = list(self.graph.out_edges(curr_node))
                visited.add(curr_node)
                for pair in neighbors:
                    # Node to be expanded is always in the second position
                    if pair[1] in visited:
                        continue
                    edge = {}
                    to_be_expanded.append(pair[1])
                    for i in range(self.graph.number_of_edges(pair[0], pair[1])):
                        nodes.add(pair[1])
                        edge = {}
                        edge["from"] = pair[0]
                        edge["to"] = pair[1]
                        edge["label"] = self.graph.edges[pair[0], pair[1], i]["label"]
                        edge["description"] = self.graph.edges[pair[0], pair[1], i]["description"]
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
        edge_dict_list = []
        for entity in seed_entities:
            edges = self.get_bfs_subgraph(entity, depth)
            edge_dict_list.extend(edges)
    
        return edge_dict_list, self.get_nodes_set(edge_dict_list)


    def extract_subgraph_edge_descriptions(self, edge_dict_list):
        edge_desc_list = []

        for edge_dict in edge_dict_list:    
            edge_desc_list.append(edge_dict["description"])

        return edge_desc_list



    def get_similarity_based_subgraph(
            self, similarity_model, question, seed_entities, depth, 
            not_to_expand_relation_labels = [], path_similarity_cutoff = 0.2
        ):

        edge_list = []
        question_embedding = similarity_model.encode(question, show_progress_bar=False)
        not_to_expand_relation_set = set(not_to_expand_relation_labels)

        # Max heap for candidate edges
        path_pool = PriorityQueue() 
        visited = set()

        for entity in seed_entities:
            if not self.graph.has_node(entity):
                continue
            visited.add(entity)
            neighbors = list(self.graph.out_edges(entity))
            for pair in neighbors:
                for i in range(self.graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = self.graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = self.graph.edges[pair[0], pair[1], i]["description"]
                    
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
            neighbors = list(nx.bfs_edges(self.graph, node_to_be_expanded, depth_limit=1))
            for pair in neighbors:
                for i in range(self.graph.number_of_edges(pair[0], pair[1])):
                    edge_dict = {}
                    edge_dict["from"] = pair[0]
                    edge_dict["to"] = pair[1]
                    edge_dict["label"] = self.graph.edges[pair[0], pair[1], i]["label"]
                    edge_dict["description"] = self.graph.edges[pair[0], pair[1], i]["description"]
                    
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
        # similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        similarity_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            cache_folder=HF_MODELS_DIR
        )
        edge_dict_list = self.get_similarity_based_subgraph(similarity_model, question, seed_entities, depth, 
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