from collections import deque
from neo4j import GraphDatabase
import csv

def scrape_subgraph_to_file(driver, seed_qids, depth, edges_file, frontier_csv, visited_csv):
    """
    BFS from seed nodes up to given depth.
    Saves:
      - edges in MetaQA format (labels only)
      - frontier nodes at max depth
      - visited nodes (all expanded so far)
    """
    visited = set(seed_qids)  # track expanded nodes
    frontier = deque([(qid, 0) for qid in seed_qids])
    edges = set()
    nodes_at_max_depth = set()

    with driver.session() as session:
        while frontier:
            qid, d = frontier.popleft()
            if d >= depth:
                # reached max depth frontier
                nodes_at_max_depth.add(qid)
                continue

            query = """
            MATCH (s {id:$qid})-[r]->(o)
            RETURN 
                s.id AS s, coalesce(s.label, s.id) AS s_lbl,
                r.prop_id AS p, coalesce(r.prop_label, r.prop_id) AS p_lbl,
                o.id AS o, coalesce(o.label, o.id) AS o_lbl
            """
            results = list(session.run(query, qid=qid))

            print(f"Found {len(results)} edges from {qid} at depth {d}")

            for record in results:
                s_lbl = record["s_lbl"]
                p_lbl = record["p_lbl"]
                o_lbl = record["o_lbl"]
                o_id  = record["o"]

                # Store edge in MetaQA format
                edge_str = f"{s_lbl}|{p_lbl}|{o_lbl}"
                edges.add(edge_str)

                # Expand further only if unseen
                if o_id not in visited:
                    visited.add(o_id)
                    frontier.append((o_id, d + 1))

            print(f"Edges collected so far: {len(edges)}")

    print(f"\nFinished scraping. Total unique edges: {len(edges)}")

    # Save edges
    with open(edges_file, "w", encoding="utf-8") as f:
        for e in sorted(edges):
            f.write(e + "\n")
    print(f"Saved {len(edges)} edges to {edges_file}")

    # Save frontier (nodes at max depth)
    with open(frontier_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        for node in sorted(nodes_at_max_depth):
            writer.writerow([node])
    print(f"Saved {len(nodes_at_max_depth)} frontier nodes to {frontier_csv}")

    # Save visited nodes
    with open(visited_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label"])
        for node in sorted(visited):
            writer.writerow([node])
    print(f"Saved {len(visited)} visited nodes to {visited_csv}")


# Example usage
seed_qids = ["Q183"]  # Germany
neo4j_uri = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "moein1378"
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

scrape_subgraph_to_file(
    driver,
    seed_qids,
    depth=2,
    edges_file="germany_subgraph_depth2.txt",
    frontier_csv="germany_depth2_frontier.csv",
    visited_csv="germany_seen_nodes.csv"
)
