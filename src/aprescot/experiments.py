import json
from typing import List, Dict, Tuple


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
            parts = line.strip().split("|")
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
    print("Subgraph Edges:", len(pred_set))
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
