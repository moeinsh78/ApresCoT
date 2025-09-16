from typing import List, Dict, Sequence
from langchain_openai import ChatOpenAI
import json
import ast
import re



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
