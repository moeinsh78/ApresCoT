import multiprocessing
import torch
from typing import Dict
from enum import Enum

# APP_NAME = "AprèsCoT"
APP_NAME = "AprèsCoT: Explaining LLM Answers with Knowledge Graphs and Chain of Thought"

DEFAULT_NUM_PERMUTATIONS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_TOKENS = 10
DEFAULT_MAX_COUNTERFACTUALS = 5
DEFAULT_TOP_K = 4


class SupportedLLM(Enum):
    GPT_35_KGRAG = "GPT-3.5 + KG RAG"
    GPT_4O_MINI_KGRAG = "GPT-4o mini + KG RAG"
    GPT_35 = "GPT-3.5"
    GPT_4O_MINI = "GPT-4o mini"


LLM_NAMES: Dict[str, str] = {
    "kg-gpt-3.5": "gpt-3.5-turbo",
    "kg-gpt-4o-mini": "gpt-4o-mini",
    "vanilla-gpt-3.5": "gpt-3.5-turbo",
    "vanilla-gpt-4o-mini": "gpt-4o-mini",
}

SYSTEM_NAMES = [
    dict(label="GPT-4o mini + KG RAG", value="kg-gpt-4o-mini"),
    dict(label="GPT-3.5 + KG RAG", value="kg-gpt-3.5"),
    dict(label="GPT-4o mini", value="vanilla-gpt-4o-mini"),
    dict(label="GPT-3.5", value="vanilla-gpt-3.5"),
    # dict(label=supported_llm.value, value=supported_llm.value)
    # for supported_llm in SupportedLLM
]


KG_OPTIONS = [
    dict(label="WikiData", value="wikidata"),
    dict(label="MetaQA Movies", value="meta-qa"),
    dict(label="UMLS Relations", value="umls"),
]

COT_COLOR_MAPPING = {
    0: "green",
    1: "light-green",
    2: "dark-green",
    3: "blue",
    4: "light-blue",
}


NUM_PARALLEL_THREADS = multiprocessing.cpu_count()
HIDDEN_STYLE = {"display": "none"}
VISIBLE_STYLE = {}
