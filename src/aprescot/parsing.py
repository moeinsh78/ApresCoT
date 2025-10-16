from typing import List, Dict, Sequence
from langchain_openai import ChatOpenAI
import json
import ast

import spacy
import nltk
from nltk import sent_tokenize
import re

# Download required NLTK data on first run
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_facts_as_triples(reasoning_text: str):
    """
    Extract simple subject-verb-object triples using spaCy dependency parsing.
    Handles compound objects and improves extraction quality.
    """
    # Handle None or non-string input
    if not reasoning_text or not isinstance(reasoning_text, str):
        return []
    
    facts = []
    seen = set()  # Track duplicates
    
    for sent in sent_tokenize(reasoning_text):
        doc = nlp(sent)
        
        # Extract subject-verb-object patterns
        for token in doc:
            if token.pos_ == "VERB":
                subjects = [child for child in token.children if child.dep_ in ("nsubj", "nsubjpass")]
                objects = [child for child in token.children if child.dep_ in ("dobj", "pobj", "attr")]
                
                # Also check for conjuncts (objects connected by 'and')
                extended_objects = objects.copy()
                for obj in objects:
                    for conj_child in obj.children:
                        if conj_child.dep_ == "conj":
                            extended_objects.append(conj_child)
                
                for subj in subjects:
                    for obj in extended_objects:
                        # Get noun phrases (limit to avoid overly complex phrases)
                        subj_phrase = " ".join([t.text for t in subj.subtree if not t.dep_ in ("punct", "cc")])
                        obj_phrase = " ".join([t.text for t in obj.subtree if not t.dep_ in ("punct", "cc")])
                        verb_phrase = token.lemma_
                        
                        # Split if object still contains commas or 'and'
                        individual_objects = re.split(r',\s*(?:and\s+)?|\s+and\s+', obj_phrase)
                        individual_objects = [o.strip() for o in individual_objects if o.strip()]
                        
                        for single_obj in individual_objects:
                            fact = f"{subj_phrase} {verb_phrase} {single_obj}."
                            
                            # Deduplicate
                            fact_normalized = fact.lower().strip()
                            if fact_normalized not in seen:
                                seen.add(fact_normalized)
                                facts.append(fact)
    
    return facts


def parse_reasoning_output(text: str):
    """
    Parse model output containing free-form reasoning followed by:
        ---FINAL ANSWERS---
        <answers>
    """

    # Normalize and strip whitespace
    text = text.replace("\r", "").strip()

    # Split reasoning and answers
    match = re.search(r"---FINAL ANSWERS---(.*)", text, re.DOTALL | re.IGNORECASE)

    if match:
        reasoning = text[: match.start()].strip()
        answers_text = match.group(1).strip()
    else:
        # Fallback: no final answers section found
        reasoning = text
        answers_text = ""

    # Parse answers: one per line or comma-separated
    answers = [
        item.strip("-• \t")
        for item in re.split(r"[\n,]+", answers_text)
        if item.strip()
    ]

    return answers, reasoning


# def parse_reasoning_output(text: str):
#     """
#     Parse model output containing the following sections:
#       ---REASONING---
#       ---PARSED ATOMIC REASONING STEPS---
#       ---FINAL ANSWERS---

#     Returns:
#         answers: list[str]
#         atomic_steps: list[str]
#     """

#     # Normalize newlines and trim whitespace
#     text = text.replace("\r", "").strip()

#     # Regex patterns for each section
#     reasoning_pattern = r"---REASONING---(.*?)---PARSED ATOMIC REASONING STEPS---"
#     atomic_pattern = r"---PARSED ATOMIC REASONING STEPS---(.*?)---FINAL ANSWERS---"
#     answers_pattern = r"---FINAL ANSWERS---(.*)"

#     reasoning_match = re.search(reasoning_pattern, text, re.DOTALL | re.IGNORECASE)
#     atomic_match = re.search(atomic_pattern, text, re.DOTALL | re.IGNORECASE)
#     answers_match = re.search(answers_pattern, text, re.DOTALL | re.IGNORECASE)

#     reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
#     atomic_text = atomic_match.group(1).strip() if atomic_match else ""
#     answers_text = answers_match.group(1).strip() if answers_match else ""

#     # Parse atomic steps: one per line (ignore empty or bullet characters)
#     atomic_steps = [
#         line.strip("-• \t")
#         for line in atomic_text.splitlines()
#         if line.strip()
#     ]

#     # Parse final answers: one per line (or comma-separated)
#     answers = [
#         item.strip("-• \t")
#         for item in re.split(r"[\n,]+", answers_text)
#         if item.strip()
#     ]

#     return answers, atomic_steps


def parse_llm_response(response: str, question: str, triples: bool = False) -> Dict[str, List[str]]:
    qa_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, model_kwargs={"response_format": {"type": "json_object"} })
    if triples:
        messages = [
        ("system", TRIPLES_PARSE_RESPONSE_INSTRUCTION),
        ("user", 'Question: \n{}\n\nModel Response: \n{}\nA model has responded in this way to the given question. Please parse out and decompose the reasoning steps of this model into independent atomic facts shaped in knowledge triples, and identify the single items of the final answer that the model has provided. Shape them all in a JSON format.\nThis would be an example of knowledge triple: <Paris, capital_of, France>'.format(question, response)),
    ]
    else: 
        messages = [
            ("system", PARSE_RESPONSE_INSTRUCTION),
            ("user", 'Question: \n{}\n\nModel Response: \n{}\nA model has responded in this way to the given question. Please parse out and decompose the reasoning steps of this model into independent atomic facts shaped in knowledge triples, and identify the single items of the final answer that the model has provided. Shape them all in a JSON format.\n'.format(question, response)),
        ]
    response = qa_llm.invoke(messages)
    response_json = json.loads(response.content)

    answer_list = [str(answer) for answer in response_json["answer"]]
    reasoning_list = [str(cot) for cot in response_json["reasoning"]]

    print("Parsed Response:", response_json)
    return answer_list, reasoning_list


def concat_triple(triple_line: str) -> str:
    """
    Turns a string such as "{'subject': 'Germany', 'predicate': 'shares_land_borders_with', 'object': 'Denmark'}"
    into "Germany shares land borders with Denmark"
    """
    triple = ast.literal_eval(triple_line)

    def _norm(s: str) -> str:
        s = s.strip()
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s)
        return s

    subj = _norm(triple.get("subject", ""))
    pred = _norm(triple.get("predicate", ""))
    obj  = _norm(triple.get("object", ""))

    return f"{subj} {pred} {obj}"



TRIPLES_PARSE_RESPONSE_INSTRUCTION = \
"""
A question has been asked from a model, and the model has provided its reasoning process and final answers to the question in plain text without any specific structure.
Your task is to parse the response, extract the reasoning and answers, structure the reasoning sentences in shape of knowledge triples (<subject, predicate, object>), and output them in a structured JSON format. 
The reasoning steps should be a collection of atomic facts, and then you should try to extract knowledge triples as if they are knowledge graph edges. 
Each reasoning step should be a clear statement that contributes to the final answer and only one piece of information. Decompose the thought process into independent sentences, each including a single fact or piece of reasoning, and then shape the triples.
Also the final answers should be a list of the answers that the model has provided in response to the question.
You will be given the question that was asked and the model's response.
Please extract the reasoning steps and final answers from the response and return them in a JSON object with two keys:
1. "reasoning": a list of atomic reasoning steps used to answer the question, in shape of knowledge triples (<subject, predicate, object>)
2. "answer": a list of final answers to the question.
"""


PARSE_RESPONSE_INSTRUCTION = \
"""
A question has been asked from a model, and the model has provided its reasoning process and final answers to the question in plain text without any specific structure.
Your task is to parse the response, extract the reasoning and answers, and output them in a structured JSON format. 
The reasoning steps should be a collection of atomic facts. Each reasoning step should be a clear statement that contributes to the final answer and only one piece of information. 
Decompose the thought process into independent sentences, each including a single fact or piece of reasoning. You should decompose the reasoning into atomic sentences.
Also the final answers should be a list of the answers that the model has provided in response to the question.
You will be given the question that was asked and the model's response.
Please extract the reasoning steps and final answers from the response and return them in a JSON object with two keys:
1. "reasoning": a list of atomic reasoning steps used to answer the question.
2. "answer": a list of final answers to the question.
"""
