import multiprocessing
import torch
from typing import Dict
from enum import Enum
from rage.llm import LLM
from rage.llm.huggingface import HuggingFaceLLM
from rage.llm.api import ChatGPTLLM

APP_NAME = "AprèsCoT"
# APP_NAME = "AprèsCoT: Explaining LLM Answers with Knowledge Graphs and Chain of Thought"

DEFAULT_NUM_PERMUTATIONS = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_TOKENS = 10
DEFAULT_MAX_COUNTERFACTUALS = 5
DEFAULT_TOP_K = 4


class SupportedLLM(Enum):
    CHATGPT_35_KGRAG = "ChatGPT 3.5 + KG RAG"
    CHATGPT_4o_MINI_KGRAG = "GPT-4o Mini + KG RAG"
    CHATGPT_35 = "ChatGPT 3.5"
    CHATGPT_4o_MINI = "GPT-4o Mini"


LLMS: Dict[str, LLM] = {
    SupportedLLM.CHATGPT_35.value: ChatGPTLLM()
}

LLM_NAMES: Dict[str, str] = {
    SupportedLLM.CHATGPT_35_KGRAG.value: "gpt-3.5-turbo",
    SupportedLLM.CHATGPT_4o_MINI_KGRAG.value: "gpt-4o-mini",
    SupportedLLM.CHATGPT_35.value: "gpt-3.5-turbo",
    SupportedLLM.CHATGPT_4o_MINI.value: "gpt-4o-mini"
}


SYSTEM_NAMES: Dict[str, str] = {
    SupportedLLM.CHATGPT_35_KGRAG.value: "kg-gpt-3.5",
    SupportedLLM.CHATGPT_4o_MINI_KGRAG.value: "kg-gpt-4o-mini",
    SupportedLLM.CHATGPT_35.value: "vanilla-gpt-3.5",
    SupportedLLM.CHATGPT_4o_MINI.value: "vanilla-gpt-4o-mini"
}


LLM_OPTIONS = [
    dict(label=supported_llm.value, value=supported_llm.value)
    for supported_llm in SupportedLLM
]

INDEX_OPTIONS = [
    dict(label="MS MARCO V1 Passage", value="msmarco-v1-passage-full"),
    dict(label="Wikipedia", value="enwiki-paragraphs"),
    dict(label="ACM Abstracts", value="cacm"),
    dict(label="Tweets", value="tweets"),
    dict(label="News Articles", value="news"),
    dict(label="Legal Clauses", value="legal_clauses"),
    dict(label="Arxiv Abstracts", value="arxiv"),
    dict(label="Blog Posts", value="blog_posts"),
    dict(label="COVID-19 Abstracts", value="cord19_abstract")
]

KAGE_INDEX_OPTIONS = [
    dict(label="MetaQA Movies", value="meta-qa"),
    dict(label="UMLS Relations", value="umls"),
]

KAGE_HOP_COUNT_OPTIONS = [
    dict(label=1, value=1),
    dict(label=2, value=2),
    dict(label=3, value=3),
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
