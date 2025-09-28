from typing import List, Dict, Tuple, Set


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

def get_nodes_and_edges_matching_gt(
    gt_file: str,
    pred_edges: List[Dict],
    undirected: bool = True,
):
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
        if undirected:
            nodes = sorted([edge["from"], edge["to"]], key=lambda x: x.lower())
            return (nodes[0], edge["label"], nodes[1])
        else:
            return (edge["from"], edge["label"], edge["to"])

    gt_set = {normalize(e) for e in gt_edges}
    pred_set = {normalize(e) for e in pred_edges}

    # true_positives = list(gt_set & pred_set)
    true_positives = [f"{edge[0]} {edge[1]} {edge[2]}" for edge in (gt_set & pred_set)]

    return [], true_positives


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
                "description": f"{head} {relation} {tail}"
            }
            edge_dict_list.append(edge)
            nodes_set.update([head, tail])
            edge_descriptions.append(edge["description"])

    return seed_nodes, nodes_set, edge_dict_list, edge_descriptions, 0
