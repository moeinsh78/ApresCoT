
def build_cyto_subgraph_elements_list(seed_nodes, nodes_set, edge_dict_list, edge_to_cot_match, answer_match_status):
    edge_elements = [] 
    graph_elements = []
    seed_nodes_set = set(str(node) for node in seed_nodes)

    for node in nodes_set:
        label = node
        node_class = "normal"
        if node in seed_nodes_set:
            node_class = "source"
        elif node in answer_match_status:
            node_class = "response"
            label = f"[A{answer_match_status[node]}] " + node

        graph_elements.append({"data": {"id": node, "label": label}, "classes": node_class})

    for i, edge_dict in enumerate(edge_dict_list):
        if (i + 1) in edge_to_cot_match:
            graph_elements.append({"data": {"source": edge_dict["from"], "target": edge_dict["to"], "weight": f"[S{edge_to_cot_match[i + 1]}] " + edge_dict["label"]}, "classes": "curved cot-edge"})
        else:
            graph_elements.append({"data": {"source": edge_dict["from"], "target": edge_dict["to"], "weight": edge_dict["label"]}, "classes": "curved"})

    return graph_elements
