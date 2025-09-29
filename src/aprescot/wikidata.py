import json
from typing import Tuple, Dict, List
import networkx as nx
import pandas as pd
import numpy as np
import requests
from queue import PriorityQueue
from openai import OpenAI
import shlex
import subprocess

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase


import time
import requests
from torch import topk

import math
import time
from typing import Dict, List, Tuple, Iterable, Optional, Set
import requests
import functools

import torch
from transformers import AutoTokenizer, AutoModel

NEO4J_PASSWORD = "moein1378"

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"

SKIP_PROPERTIES = {"P31","P279","P21","P27","P101","P106"} 

DEFAULT_SCORER = "drt/srtk-scorer"  # pretrained, no training needed

HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "ApresCoT-Subgraph/1.0 (mailto:you@example.com)"
}

# WikiData Properties to Skip

ALWAYS_SKIP = {
    "P18", "P154", "P373", "P41", "P109", "P94",
    "P31", "P279", "P1343", "P143", "P4656",
    "P214", "P227", "P244", "P268", "P269", "P345",
    "P646", "P950", "P213", "P691", "P245", "P2148",
    "P17", "P131", "P625", "P856", "P6375", "P281",
    "P1476", "P577", "P571", "P576"
}

NOT_TO_SKIP = {
    "person": {},
    "film": {},
    "country": {},
}

# NOT_TO_SKIP = {
#     "person": {"P569", "P570"},
#     "film": {"P161", "P57"},
#     "country": {"P36", "P37"},
# }



# FILM_MINIMAL = {"P31","P279","P57","P58","P161","P162","P344","P1040","P86","P272","P750","P136","P921","P364","P495","P915","P840","P361","P527","P179"}


# TENNIS_WHITELIST = {
#     # Core typing
#     "P31",   # instance of
#     "P279",  # subclass of

#     # Sports-specific
#     "P641",  # sport
#     "P1344", # participant in
#     "P1346", # winner
#     "P2522", # victory
#     "P1340", # points/score in ranking systems
#     "P1350", # number of matches played
#     "P1351", # number of wins
#     "P1352", # ranking
#     "P1353", # wins at a specific tournament

#     # Awards & achievements
#     "P166",  # award received

#     # Personal data / career context
#     "P27",   # country of citizenship
#     "P19",   # place of birth
#     "P106",  # occupation
#     "P54",   # member of sports team
#     "P569",  # date of birth
#     "P570",  # date of death
# }


# MINIMAL_COUNTRIES = {"P17","P47","P31","P279","P361","P527","P921","P131","P17","P495"}


# BALANCED = {
#     "P31","P279","P361","P527","P921",
#     "P131","P17","P495","P276",
#     "P170","P57","P58","P86","P50","P161","P1029",
#     "P136","P2283",
#     "P571","P577",
#     "P463","P102","P108","P39","P106",
#     "P530","P1376","P155","P156","P641"
# }


class WikiDataKnowledgeGraph:
    def search_wikidata_entity(self, query, language="en"):
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": language,
            "uselang": language,
            "type": "item",
            "search": query,
            "limit": 5,
        }
        headers = self.headers

        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        results = r.json().get("search", [])

        if results:
            top = results[0]
            return top["id"], top.get("label"), top.get("description", "")
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

    def __init__(
        self,
        sparql_endpoint: str = WIKIDATA_SPARQL,
        scorer_model: str = DEFAULT_SCORER,
        device: Optional[str] = None,
        user_agent: str = "ApresCoT/1.0 (mailto:moeiiinsh@gmail.com)",
        req_timeout_s: int = 60,
        polite_qps: float = 1.0,
        use_local_db: bool = False,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = NEO4J_PASSWORD,
    ):
        # existing init...
        self.endpoint = sparql_endpoint
        self.headers = {"Accept": "application/sparql-results+json", "User-Agent": user_agent}
        self.req_timeout_s = req_timeout_s
        self.sleep_between = 1.0 / max(polite_qps, 1e-6)

        # encoder setup unchanged...
        self.tokenizer = AutoTokenizer.from_pretrained(scorer_model)
        self.model = AutoModel.from_pretrained(scorer_model)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model.to(self.device).eval()

        # caches
        self._label_cache = {}
        self._neighbors_cache = {}

        # new: Neo4j driver
        self.use_local_db = use_local_db
        if self.use_local_db:
            self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        else:
            self.driver = None



    def retrieve_with_srtk_style(
        self,
        question: str,
        seed_qids: List[str],
        *,
        max_hops,
        beam_size,
        per_pred_cap,
        total_cap_per_node,
        max_nodes,
        compare_to_hypothetical_answer: bool = True,
        add_labels: bool = True,
    ):
        """
        Returns:
          {
            "question": str,
            "seeds": [Q...],
            "triples": [{"s":"Q..","p":"P..","o":"Q.."|literal}, ...],
            "labels": { "Q..": "Label", "P..": "prop label", ... }
          }
        """
        start = time.perf_counter()
        if compare_to_hypothetical_answer:
            hypothetical_answer = generate_hypothetical_answer(question)
            print("Hypothetical Answer:", hypothetical_answer)
            q_vec = self._encode_text(hypothetical_answer)
        else:
            q_vec = self._encode_text(question)

        print("## TIME ## after generating hyde:", time.perf_counter() - start)

        # subgraph state
        triples: List[Dict[str, str]] = []
        seen_triple: Set[Tuple[str, str, str]] = set()
        seen_node: Set[str] = set(seed_qids)
        frontier: Set[str] = set(seed_qids)

        print("Retrieving: ", frontier, seed_qids)

        for hop in range(max_hops):
            # gather candidate edges from frontier
            candidates: List[Tuple[float, Dict[str, str]]] = []
            for qid in list(frontier):
                print("Adding neighbors for:", qid)
                if self.use_local_db:
                    neigh = self._neighbors_local_db(qid, direction="out",
                                            per_pred_cap=per_pred_cap,
                                            total_cap=total_cap_per_node)
                    
                    print("## TIME ## after finding", qid, "'s neighbours:", time.perf_counter() - start)

                else:
                    neigh = self._neighbors_api(qid, direction="out",
                                                per_pred_cap=per_pred_cap,
                                                total_cap=total_cap_per_node)
                for t in neigh:
                    key = (t["s"], t["p"], t["o"])
                    if key in seen_triple:
                        continue
                    # textualize edge for scoring
                    text = self._edge_textualization(t)
                    # print("Edge Textualization => Triple: ", key, " -- Text: ", text)
                    score = self._score_text(q_vec, text)
                    # print("Edge score: ", score)
                    candidates.append((score, t))
                    
                print("## TIME ## scoring candidates for", qid, ":", time.perf_counter() - start)


            if not candidates:
                break

            # beam: keep only top scoring edges this hop
            candidates.sort(key=lambda x: x[0], reverse=True)
            keep = candidates[:beam_size]

            # add to graph + build next frontier
            new_frontier: Set[str] = set()
            for _, t in keep:
                key = (t["s"], t["p"], t["o"])
                if key in seen_triple:
                    continue
                triples.append(t)
                seen_triple.add(key)
                if isinstance(t["o"], str) and t["o"].startswith("Q"):
                    new_frontier.add(t["o"])
                    seen_node.add(t["o"])
                if isinstance(t["s"], str) and t["s"].startswith("Q"):
                    seen_node.add(t["s"])

                if len(triples) >= max_nodes:
                    break

            frontier = new_frontier
            if len(triples) >= max_nodes or not frontier:
                break

        result = {"question": question, "seeds": seed_qids, "triples": triples, "labels": {}}

        if add_labels and triples:
            qids, pids = self._collect_ids(triples)
            qids.update(seed_qids)
            labels = self._fetch_labels_bulk(list(qids), list(pids))
            result["labels"] = labels

        return self.srtk_output_to_labeled_graph(result)
    
    def srtk_output_to_labeled_graph(self, res: Dict):
        """
        Transform the srtk_style output into:
        seed_labels, nodes_set, edge_dict_list, edge_descriptions
        without making any additional API calls.
        """
        labels = res.get("labels", {})
        triples = res.get("triples", [])
        seeds = res.get("seeds", [])

        def to_lbl(x: str) -> str:
            # map Q*/P* to label if we have it; otherwise fall back to on-demand label()
            if isinstance(x, str) and (x.startswith("Q") or x.startswith("P")):
                return labels.get(x) or self._label(x)
            return x

        # Seed labels
        seed_labels = [to_lbl(s) for s in seeds]

        # Node labels (collect all subjects + objects)
        node_labels = set()
        for t in triples:
            node_labels.add(to_lbl(t["s"]))
            node_labels.add(to_lbl(t["o"]))

        # Edges with labels + human-readable description
        edge_dict_list = []
        for t in triples:
            s_lbl = to_lbl(t["s"])
            p_lbl = to_lbl(t["p"])
            o_lbl = to_lbl(t["o"])
            edge_dict_list.append({
                "from": s_lbl,
                "to": o_lbl,
                "label": p_lbl,
                "description": f"{s_lbl} - {p_lbl} - {o_lbl}",
            })

        edge_descriptions = [e["description"] for e in edge_dict_list]

        return seed_labels, node_labels, edge_dict_list, edge_descriptions

    # ------------------ scoring ------------------

    @torch.inference_mode()
    def _encode_text(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        # mean-pool (works well for many sentence encoders)
        last_hidden = outputs.last_hidden_state  # [1, L, H]
        attn_mask = tokens["attention_mask"].unsqueeze(-1)  # [1, L, 1]
        masked = last_hidden * attn_mask
        emb = masked.sum(dim=1) / (attn_mask.sum(dim=1) + 1e-9)
        return emb[0].detach().cpu()

    def _score_text(self, q_vec: torch.Tensor, text: str) -> float:
        v = self._encode_text(text)
        return _cosine_sim(q_vec, v)

    def _neighbors_local_db(self, qid, direction, per_pred_cap, total_cap, domain = None):
        """Fetch neighbors from Neo4j"""
        print(f"Fetching neighbors from local Neo4j DB for {qid}"),
        triples = []
        skip_props = self._get_skip_props(domain)

        with self.driver.session() as session:
            if direction in ("out", "both"):
                q = """
                MATCH (s {id:$qid})-[r]->(o)
                WHERE NOT r.prop_id IN $skip_props
                RETURN s.id AS s, r.prop_id AS p, o.id AS o
                """
                res = session.run(q, qid=qid,
                                skip_props=list(skip_props),
                                # limit=total_cap
                                )
                triples.extend([dict(record) for record in res])

            if direction in ("in", "both"):
                q = """
                MATCH (s)-[r]->(o {id:$qid})
                WHERE NOT r.prop_id IN $skip_props
                RETURN s.id AS s, r.prop_id AS p, o.id AS o
                """
                res = session.run(q, qid=qid,
                                skip_props=list(skip_props),
                                # limit=total_cap
                                )
                triples.extend([dict(record) for record in res])

        print("Number of edges found for", qid, ":", len(triples))
        # Apply per-predicate cap
        limited, counts = [], {}
        for t in triples:
            key2 = (t["s"], t["p"])
            counts[key2] = counts.get(key2, 0) + 1
            if counts[key2] <= per_pred_cap:
                limited.append(t)
        print("Number of edges found for", qid, " after applying per-pred cap:", len(limited))
        return limited

    # def _neighbors_local_db(self, qid, direction, per_pred_cap, total_cap):
    #     """Fetch neighbors from Neo4j"""
    #     print(f"Fetching neighbors from local Neo4j DB for {qid}"),
    #     triples = []
    #     with self.driver.session() as session:
    #         if direction in ("out", "both"):
    #             q = """
    #             MATCH (s {id:$qid})-[r]->(o)
    #             RETURN s.id AS s, r.prop_id AS p, o.id AS o
    #             LIMIT $limit
    #             """
    #             res = session.run(q, qid=qid, limit=total_cap)

    #             triples.extend([dict(record) for record in res])

    #         if direction in ("in", "both"):
    #             q = """
    #             MATCH (s)-[r]->(o {id:$qid})
    #             RETURN s.id AS s, r.prop_id AS p, o.id AS o
    #             LIMIT $limit
    #             """
    #             res = session.run(q, qid=qid, limit=total_cap)

    #             triples.extend([dict(record) for record in res])

    #     print("Number of edges found for", qid, ":", len(triples))
    #     # cap per (s,p)
    #     limited, counts = [], {}
    #     for t in triples:
    #         key2 = (t["s"], t["p"])
    #         counts[key2] = counts.get(key2, 0) + 1
    #         if counts[key2] <= per_pred_cap:
    #             limited.append(t)
    #     print("Number of edges found for", qid, " after applying per-pred cap:", len(limited))
    #     return limited


    def _neighbors_api(self, qid, direction, per_pred_cap, total_cap) -> List[Dict[str, str]]:
        print(f"Fetching neighbors from Wikidata API for {qid}")
        key = (qid, f"{direction}:{per_pred_cap}:{total_cap}")
        if key in self._neighbors_cache:
            return self._neighbors_cache[key]

        # Build a NOT IN(...) clause for wdt: predicates we want to skip
        if SKIP_PROPERTIES:
            skip_list = ", ".join(f"wdt:{p}" for p in SKIP_PROPERTIES)
            filter_skip = f"FILTER(?p NOT IN ({skip_list}))"
        else:
            filter_skip = ""

        queries = []
        if direction in ("out", "both"):
            queries.append(f"""
            SELECT ?s ?p ?o WHERE {{
            VALUES ?s {{ wd:{qid} }}
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), STR(wdt:)))
            {filter_skip}
            }} LIMIT {total_cap}
            """)
        if direction in ("in", "both"):
            queries.append(f"""
            SELECT ?s ?p ?o WHERE {{
            VALUES ?o {{ wd:{qid} }}
            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), STR(wdt:)))
            {filter_skip}
            }} LIMIT {total_cap}
            """)

        triples: List[Dict[str, str]] = []
        for q in queries:
            time.sleep(self.sleep_between)
            resp = requests.get(self.endpoint, params={"query": q}, headers=self.headers, timeout=self.req_timeout_s)
            resp.raise_for_status()
            for b in resp.json().get("results", {}).get("bindings", []):
                s_uri = b["s"]["value"]; p_uri = b["p"]["value"]; o_val = b["o"]["value"]
                s_id = s_uri.rsplit("/", 1)[-1]
                p_id = p_uri.rsplit("/", 1)[-1]
                # object may be entity or literal
                o_id = o_val.rsplit("/", 1)[-1] if o_val.startswith("http://www.wikidata.org/entity/") else o_val
                if p_id.startswith("P"):
                    triples.append({"s": s_id, "p": p_id, "o": o_id})

        # Belt & suspenders: drop any skipped props that slipped through
        if SKIP_PROPERTIES:
            triples = [t for t in triples if t["p"] not in SKIP_PROPERTIES]

        # cap per (s,p)
        limited: List[Dict[str, str]] = []
        counts: Dict[Tuple[str, str], int] = {}
        for t in triples:
            key2 = (t["s"], t["p"])
            counts[key2] = counts.get(key2, 0) + 1
            if counts[key2] <= per_pred_cap:
                limited.append(t)

        self._neighbors_cache[key] = limited
        return limited

    def _get_skip_props(self, domain: Optional[str] = None) -> set:
        skip_props = ALWAYS_SKIP.copy()
        if domain and domain in NOT_TO_SKIP:
            skip_props -= NOT_TO_SKIP[domain]  # rescue some props
        return skip_props

    # ------------------ textualization + labels ------------------

    def _edge_textualization(self, t: Dict[str, str]) -> str:
        s = t["s"]; p = t["p"]; o = t["o"]
        s_lbl = self._label(s)
        p_lbl = self._label(p)
        if isinstance(o, str) and o.startswith("Q"):
            o_lbl = self._label(o)
            return f"{s_lbl} — {p_lbl} — {o_lbl}"
        else:
            return f"{s_lbl} — {p_lbl} — {o}"

    def _label(self, pid_or_qid: str) -> str:
        if pid_or_qid in self._label_cache:
            return self._label_cache[pid_or_qid]
        lab = self._fetch_one_label(pid_or_qid)
        self._label_cache[pid_or_qid] = lab
        return lab

    def _fetch_one_label(self, pid_or_qid: str) -> str:
        is_prop = pid_or_qid.startswith("P")
        var = "?prop" if is_prop else "?item"
        lab = "?propLabel" if is_prop else "?itemLabel"
        values = f"(wd:{pid_or_qid})"
        q = f"""
        SELECT {var} {lab} WHERE {{
          VALUES ({var}) {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}"""
        time.sleep(self.sleep_between)
        try:
            r = requests.get(self.endpoint, params={"query": q}, headers=self.headers, timeout=self.req_timeout_s)
            r.raise_for_status()
            bindings = r.json().get("results", {}).get("bindings", [])
            if bindings:
                return bindings[0].get(lab[1:], {}).get("value", pid_or_qid)
        except Exception:
            pass
        return pid_or_qid

    def _collect_ids(self, triples: List[Dict[str, str]]) -> Tuple[Set[str], Set[str]]:
        qids, pids = set(), set()
        for t in triples:
            s, p, o = t["s"], t["p"], t["o"]
            if s.startswith("Q"): qids.add(s)
            if p.startswith("P"): pids.add(p)
            if isinstance(o, str) and o.startswith("Q"): qids.add(o)
        return qids, pids

    def _fetch_labels_bulk(self, qids: List[str], pids: List[str]) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        def chunks(xs, n): 
            for i in range(0, len(xs), n): 
                yield xs[i:i+n]

        # entities
        for part in chunks(qids, 150):
            values = " ".join(f"(wd:{q})" for q in part)
            q = f"""
            SELECT ?item ?itemLabel WHERE {{
              VALUES (?item) {{ {values} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}"""
            time.sleep(self.sleep_between)
            r = requests.get(self.endpoint, params={"query": q}, headers=self.headers, timeout=self.req_timeout_s)
            r.raise_for_status()
            for b in r.json().get("results", {}).get("bindings", []):
                qid = b["item"]["value"].rsplit("/", 1)[-1]
                lab = b.get("itemLabel", {}).get("value")
                if lab: labels[qid] = lab

        # properties
        for part in chunks(pids, 150):
            values = " ".join(f"(wd:{p})" for p in part)
            q = f"""
            SELECT ?prop ?propLabel WHERE {{
              VALUES (?prop) {{ {values} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}"""
            time.sleep(self.sleep_between)
            r = requests.get(self.endpoint, params={"query": q}, headers=self.headers, timeout=self.req_timeout_s)
            r.raise_for_status()
            for b in r.json().get("results", {}).get("bindings", []):
                pid = b["prop"]["value"].rsplit("/", 1)[-1]
                lab = b.get("propLabel", {}).get("value")
                if lab: labels[pid] = lab

        return labels

    # def run_sparql(self, query, retry=3, backoff=0.4):
    #     last_err = None
    #     for i in range(retry):
    #         try:
    #             r = requests.get(WDQS, params={"query": query}, headers=HEADERS, timeout=60)
    #             if r.status_code == 200:
    #                 return r.json()["results"]["bindings"]
    #             last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    #         except Exception as e:
    #             last_err = e
    #         time.sleep(backoff * (i + 1))
    #     raise last_err

    # def _qid(self, uri: str) -> str:
    #     return uri.rsplit("/", 1)[-1]

    # def _pid(self, uri: str) -> str:
    #     return uri.rsplit("/", 1)[-1]

    # def one_hop(self, seed_q, limit=200, p_whitelist=None):
    #     # Outgoing: seed wdt:P ?o (entity)
    #     q_out = f"""
    #     SELECT ?p ?o WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?seed ?p ?o .
    #     FILTER(STRSTARTS(STR(?p), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?o), STR(wd:)))
    #     }} LIMIT {limit*2}
    #     """
    #     # Incoming: ?s wdt:P seed (entity)
    #     q_in = f"""
    #     SELECT ?s ?p WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?s ?p ?seed .
    #     FILTER(STRSTARTS(STR(?p), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?s), STR(wd:)))
    #     }} LIMIT {limit*2}
    #     """

    #     out = self.run_sparql(q_out)
    #     inn = self.run_sparql(q_in)

    #     out_edges, in_edges = [], []
    #     for b in out:
    #         p = self._pid(b["p"]["value"])
    #         o = self._qid(b["o"]["value"])
    #         if (p_whitelist is None) or (p in p_whitelist):
    #             out_edges.append((seed_q, p, o))
    #             if len(out_edges) >= limit: break

    #     for b in inn:
    #         s = self._qid(b["s"]["value"])
    #         p = self._pid(b["p"]["value"])
    #         if (p_whitelist is None) or (p in p_whitelist):
    #             in_edges.append((s, p, seed_q))
    #             if len(in_edges) >= limit: break

    #     return out_edges, in_edges

    # def two_hop(self, seed_q, per_path_limit=400, p_whitelist=None):
    #     # Outgoing→Outgoing: seed -p1-> m -p2-> x
    #     q_oo = f"""
    #     SELECT ?p1 ?m ?p2 ?x WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?seed ?p1 ?m .
    #     ?m ?p2 ?x .
    #     FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?m), STR(wd:)))
    #     FILTER(STRSTARTS(STR(?x), STR(wd:)))
    #     }} LIMIT {per_path_limit*2}
    #     """
    #     # Incoming→Incoming: s -p1-> seed ; y -p2-> s
    #     q_ii = f"""
    #     SELECT ?y ?p2 ?s ?p1 WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?s ?p1 ?seed .
    #     ?y ?p2 ?s .
    #     FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?s), STR(wd:)))
    #     FILTER(STRSTARTS(STR(?y), STR(wd:)))
    #     }} LIMIT {per_path_limit*2}
    #     """
    #     # Outgoing→Incoming: seed -p1-> m ; y -p2-> m
    #     q_oi = f"""
    #     SELECT ?p1 ?m ?y ?p2 WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?seed ?p1 ?m .
    #     ?y ?p2 ?m .
    #     FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?m), STR(wd:)))
    #     FILTER(STRSTARTS(STR(?y), STR(wd:)))
    #     }} LIMIT {per_path_limit*2}
    #     """
    #     # Incoming→Outgoing: s -p1-> seed ; s -p2-> x
    #     q_io = f"""
    #     SELECT ?s ?p1 ?p2 ?x WHERE {{
    #     VALUES ?seed {{ wd:{seed_q} }}
    #     ?s ?p1 ?seed .
    #     ?s ?p2 ?x .
    #     FILTER(STRSTARTS(STR(?p1), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?p2), STR(wdt:)))
    #     FILTER(STRSTARTS(STR(?s), STR(wd:)))
    #     FILTER(STRSTARTS(STR(?x), STR(wd:)))
    #     }} LIMIT {per_path_limit*2}
    #     """

    #     results = {
    #         "oo": self.run_sparql(q_oo),
    #         "ii": self.run_sparql(q_ii),
    #         "oi": self.run_sparql(q_oi),
    #         "io": self.run_sparql(q_io),
    #     }

    #     edges = []
    #     def keep_p(p): return (p_whitelist is None) or (p in p_whitelist)

    #     for b in results["oo"]:
    #         p1 = self._pid(b["p1"]["value"]); m = self._qid(b["m"]["value"])
    #         p2 = self._pid(b["p2"]["value"]); x = self._qid(b["x"]["value"])
    #         if keep_p(p1) and keep_p(p2):
    #             edges.append((seed_q, p1, m))
    #             edges.append((m, p2, x))
    #     for b in results["ii"]:
    #         y = self._qid(b["y"]["value"]); p2 = self._pid(b["p2"]["value"])
    #         s = self._qid(b["s"]["value"]); p1 = self._pid(b["p1"]["value"])
    #         if keep_p(p1) and keep_p(p2):
    #             edges.append((s, p1, seed_q))
    #             edges.append((y, p2, s))
    #     for b in results["oi"]:
    #         p1 = self._pid(b["p1"]["value"]); m = self._qid(b["m"]["value"])
    #         y  = self._qid(b["y"]["value"]);  p2 = self._pid(b["p2"]["value"])
    #         if keep_p(p1) and keep_p(p2):
    #             edges.append((seed_q, p1, m))
    #             edges.append((y, p2, m))
    #     for b in results["io"]:
    #         s = self._qid(b["s"]["value"]); p1 = self._pid(b["p1"]["value"])
    #         p2 = self._pid(b["p2"]["value"]); x = self._qid(b["x"]["value"])
    #         if keep_p(p1) and keep_p(p2):
    #             edges.append((s, p1, seed_q))
    #             edges.append((s, p2, x))

    #     # Dedup
    #     edges = list(dict.fromkeys(edges))
    #     return edges

    # def fetch_labels(self, qids, pids, lang="en"):
    #     labels = {}
    #     qids = list(set(qids))
    #     pids = list(set(pids))

    #     # Entity labels
    #     for i in range(0, len(qids), 200):
    #         vals = " ".join(f"wd:{q}" for q in qids[i:i+200])
    #         q = f"""
    #         SELECT ?e ?eLabel WHERE {{
    #         VALUES ?e {{ {vals} }}
    #         SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
    #         }}
    #         """
    #         for b in self.run_sparql(q):
    #             labels[self._qid(b["e"]["value"])] = b.get("eLabel", {}).get("value", "")

    #     # Property labels
    #     for i in range(0, len(pids), 200):
    #         vals = " ".join(f"wd:{p}" for p in pids[i:i+200])
    #         q = f"""
    #         SELECT ?p ?pLabel WHERE {{
    #         VALUES ?p {{ {vals} }}
    #         SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{lang}". }}
    #         }}
    #         """
    #         for b in self.run_sparql(q):
    #             labels[self._pid(b["p"]["value"])] = b.get("pLabel", {}).get("value", "")
            
    #     return labels

    # def extract_subgraph_two_hops(
    #     self,
    #     seed_qids,
    #     one_hop_limit=200,
    #     two_hop_limit=400,
    #     p_whitelist=None,
    #     label_lang="en",
    #     total_edge_cap=8000
    # ):
    #     nodes = set(seed_qids)
    #     edges = []

    #     # 1-hop (both directions)
    #     for q in seed_qids:
    #         out_e, in_e = self.one_hop(q, limit=one_hop_limit, p_whitelist=p_whitelist)
    #         edges.extend(out_e); edges.extend(in_e)
    #         for s,p,o in out_e: nodes.update([s,o])
    #         for s,p,o in in_e:  nodes.update([s,o])

    #         if len(edges) >= total_edge_cap: break

    #     # 2-hop (four patterns) rooted at each seed
    #     if len(edges) < total_edge_cap:
    #         for q in seed_qids:
    #             e2 = self.two_hop(q, per_path_limit=two_hop_limit, p_whitelist=p_whitelist)
    #             edges.extend(e2)
    #             for s,p,o in e2: nodes.update([s,o])
    #             if len(edges) >= total_edge_cap: break

    #     # Dedup & trim
    #     edges = list(dict.fromkeys(edges))[:total_edge_cap]

    #     # Gather labels
    #     qids = {n for n in nodes if n.startswith("Q")}
    #     pids = {p for _,p,_ in edges}
    #     labels = self.fetch_labels(list(qids), list(pids), lang=label_lang)

    #     edges = [(s, labels.get(p, p), o) for (s, p, o) in edges]

    #     return {
    #         "nodes": list(nodes),
    #         "edges": edges,          # (subjectQ, predicateP, objectQ)
    #         "labels": labels,        # Q/P -> label
    #         "meta": {
    #             "seeds": seed_qids,
    #             "one_hop_limit": one_hop_limit,
    #             "two_hop_limit": two_hop_limit,
    #             "total_edge_cap": total_edge_cap,
    #             "predicate_whitelist": sorted(list(p_whitelist)) if p_whitelist else None
    #         }
    #     }
    
    # def extract_relevant_subgraph(self, seed_qids: List[str]) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]], List[str]]:
    #     p_whitelist = FILM_MINIMAL
    #     sg = self.extract_subgraph_two_hops(
    #         seed_qids,
    #         one_hop_limit=80,
    #         two_hop_limit=500,
    #         p_whitelist=p_whitelist,
    #         total_edge_cap=2000
    #     )
    #     print(f"Nodes: {len(sg['nodes'])}, Edges: {len(sg['edges'])}")
    #     L = sg["labels"]
    #     edges_labeled = [
    #         (L.get(s, s), L.get(p, p), L.get(o, o))
    #         for (s, p, o) in sg["edges"]
    #     ]

    #     nodes_labeled = [
    #         (L.get(n, n), n)[0] for n in sg["nodes"]
    #     ]
    #     nodes_set= set(nodes_labeled)

    #     edge_dict_list = [
    #         {
    #             "from": s,
    #             "to": o,
    #             "label": p,
    #             "description": f"{s} {p} {o}"
    #         }
    #         for (s, p, o) in edges_labeled
    #     ]
        
    #     edge_descriptions = [edge_dict["description"] for edge_dict in edge_dict_list]

    #     for s, p, o in edges_labeled[:200]:
    #         print(f"({s}) -[{p}]-> ({o})")

    #     return nodes_set, edge_dict_list, edge_descriptions





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

def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(p=2) + 1e-12)
    b = b / (b.norm(p=2) + 1e-12)
    return float((a @ b).item())


@functools.lru_cache(maxsize=8192)
def _label_cache(kind: str, pid_or_qid: str) -> str:
    # placeholder for lru cache signature
    return ""




def generate_hypothetical_answer(question: str, model_name="gpt-4o-mini", temperature=0.7, max_tokens=512, n=1) -> str:
    client = OpenAI()
    result = client.chat.completions.create(
        messages=[{"role":"user", "content": HYPOTHETICAL_ANSWER_PROMPT.format(question)}],
        model=model_name,
        max_completion_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    return result.choices[0].message.content


HYPOTHETICAL_ANSWER_PROMPT = """Please write a passage to answer the question.
Question: {}
Passage:"""