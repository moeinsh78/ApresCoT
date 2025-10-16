import csv
from collections import deque
from neo4j import GraphDatabase

def expand_to_depth3(driver, frontier_csv, visited_csv, prev_edges_file, new_edges_file):
    """
    Expand the depth-2 frontier one more layer (to depth 3),
    merge new edges with the previous edge file, and save combined edges.
    """
    print("=== Starting depth-3 expansion ===")

    # Load visited nodes
    visited = set()
    with open(visited_csv, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            visited.add(line.strip())
    print(f"Loaded {len(visited)} visited nodes from {visited_csv}")

    # Load frontier nodes
    frontier = deque()
    with open(frontier_csv, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            frontier.append(line.strip())
    print(f"Loaded {len(frontier)} frontier nodes from {frontier_csv}")

    # Load previous edges
    edges = set()
    with open(prev_edges_file, "r", encoding="utf-8") as f:
        for line in f:
            edges.add(line.strip())
    print(f"Loaded {len(edges)} edges from {prev_edges_file}")

    new_edges = set()

    with driver.session() as session:
        while frontier:
            qid = frontier.popleft()
            print(f"\nExpanding node {qid} ...")

            query = """
            MATCH (s {id:$qid})-[r]->(o)
            RETURN 
                coalesce(s.label, s.id) AS s_lbl,
                coalesce(r.prop_label, r.prop_id) AS p_lbl,
                coalesce(o.label, o.id) AS o_lbl,
                o.id AS o_id
            """
            results = list(session.run(query, qid=qid))
            print(f"  Found {len(results)} edges from {qid}")

            for record in results:
                s_lbl = record["s_lbl"]
                p_lbl = record["p_lbl"]
                o_lbl = record["o_lbl"]
                o_id  = record["o_id"]

                edge_str = f"{s_lbl}|{p_lbl}|{o_lbl}"
                if edge_str not in edges:
                    new_edges.add(edge_str)
                    edges.add(edge_str)

    print(f"\n=== Expansion complete ===")
    print(f"New edges discovered at depth-3: {len(new_edges)}")
    print(f"Total edges (including previous): {len(edges)}")

    # Save combined edges
    with open(new_edges_file, "w", encoding="utf-8") as f:
        for e in sorted(edges):
            f.write(e + "\n")

    print(f"Saved merged edge set to {new_edges_file}")


neo4j_uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(neo4j_uri, auth=("neo4j", "moein1378"))

expand_to_depth3(
    driver,
    frontier_csv="germany_depth2_frontier.csv",
    visited_csv="germany_seen_nodes.csv",
    prev_edges_file="germany_subgraph_depth2.txt",
    new_edges_file="germany_subgraph_depth3.txt"
)
