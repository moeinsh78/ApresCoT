from typing import List, Dict, Sequence
import re


def build_rag_qa_prompt(edge_descriptions: List[str], question: str):
    question = re.sub(r'[\[\]]', '\"', question)
    edge_description_str = "\n".join(edge_descriptions)

    prompt ="""<Context>
{context}
</Context>


<Question>
{question}
</Question>

Let’s answer the Question by reasoning step by step.
""".format(context=edge_description_str, question=question)
    return prompt

def build_regular_rag_qa_prompt(edge_descriptions: List[str], question: str):
    question = re.sub(r'[\[\]]', '\"', question)
    edge_description_str = "\n".join(edge_descriptions)


    prompt = """# CONTEXT:
{}

# QUERY:
{}

Please provide your answer to this question based on the provided context, and explain your reasoning step by step by pointing out the information you used.
""".format(edge_description_str, question)
    return prompt



def create_prompt(question: str, kg_name: str, rag: bool, llm: str, edge_descriptions: List[str], new_reasoning: bool = True, extension: bool = False):
    if llm == "o3-mini":
        return "", f"{question}\nAnswer this question by thinking step by step, and provide your reasoning process.\n"
    if new_reasoning:
        if kg_name == "wikidata":
            return WIKIDATA_REASONING_INSTRUCTION, f"# QUERY:\n{question}\n\nPlease provide your answer to this question and explain your reasoning step by step.\n"
        if rag:
            return MOVIES_REGULAR_REASONING_INSTRUCTION, build_regular_rag_qa_prompt(edge_descriptions, question) 
        else:
            return MOVIES_REGULAR_REASONING_INSTRUCTION, f"# QUERY:\n{question}\n\nPlease provide your answer to this question and explain your reasoning step by step.\n"
    if extension:
        if rag:
            return EXTENSION_RAG_FREE_FORM_INSTRUCTION, EXTENSION_RAG_PROMPT.format(context="\n".join(edge_descriptions), question=question)
        else:
            return EXTENSION_FREE_FORM_INSTRUCTION, EXTENSION_VANILLA_PROMPT.format(question=question)
    if rag:
        return KG_RAG_COT_INSTRUCTION, build_rag_qa_prompt(edge_descriptions, question)
    else:
        return VANILLA_COT_INSTRUCTION, f"# QUERY:\n{question}\n\nLet's think step by step. Please provide your reasoning steps and answers clearly. \n"

EXTENSION_FREE_FORM_INSTRUCTION = \
"""You are a reasoning assistant for question answering.

For each question, follow these steps:

1. Think step by step, showing your reasoning in natural language in the REASONING section.

2. After reasoning, rewrite your reasoning steps by extracting all the atomic factual statement you used or implied.
   - list the sentences in the PARSED ATOMIC REASONING STEPS section.
   - Each atomic fact must express exactly one piece of information.
   - Avoid combining facts with words like "and", "also", or "which", or separating them with commas.

3. Then extract from your reasoning all the final answers you found by listing them in the FINAL ANSWERS section.

4. Use the following exact format:

---REASONING---
<your detailed reasoning here>

---PARSED ATOMIC REASONING STEPS---
<each piece of information on a new line>

---FINAL ANSWERS---
<list of final answers each in a separate line>

Do not include anything outside these three sections.
"""


EXTENSION_RAG_FREE_FORM_INSTRUCTION = \
"""You are a reasoning assistant for question answering.
You will be given a question and a block of context information retrieved from a knowledge base.

For each question, follow these steps:

1. Read the Context carefully and write down your reasoning step by step in the REASONING section.

2. Rewrite your reasoning steps by extracting all the atomic factual statements in the Context that you used or implied.
   - List these in the PARSED ATOMIC REASONING STEPS section.
   - Each atomic fact must express exactly one piece of information.
   - Avoid combining facts with words like "and", "also", or "which", or separating them with commas.

3. Then extract from your reasoning all final answers you found and list them in the
   FINAL ANSWERS section (one per line). 

4. Use the following exact format:

---REASONING---
<your detailed reasoning here>

---PARSED ATOMIC REASONING STEPS---
<each statement from the context on a new line>

---FINAL ANSWERS---
<list of final answers each in a separate line>

Do not include anything outside these three sections.
"""


EXTENSION_INSTRUCTIONS = \
"""You are a reasoning assistant for question answering.

You must always produce output in JSON format with the following structure:
{
  "Answers": [ ... ],
  "Chain of Thought": [ ... ]
}

Rules:
- "Answers" is a list of concise entities or short phrases that directly answer the question. Deduplicate items.
- "Chain of Thought" is a list of *atomic reasoning steps*. Each step must express exactly one fact or inference.
- Do NOT merge multiple facts in one sentence. Avoid "and", "also", or "which" to join facts.
- Use short, declarative sentences for each step.
- Think step by step before writing the JSON. Your reasoning steps must appear in "Chain of Thought" as individual atomic facts.
"""


EXTENSION_RAG_INSTRUCTIONS = \
"""You are a reasoning assistant for question answering.

You must always produce output in JSON format with the following structure:
{
  "Answers": [ ... ],
  "Chain of Thought": [ ... ]
}

Rules:
- "Answers" is a list of concise entities or short phrases that directly answer the question. Deduplicate items.
- "Chain of Thought" is a list of *atomic reasoning steps*. Each step must express exactly one fact or inference.
- Do NOT merge multiple facts in one sentence. Avoid "and", "also", or "which" to join facts.
- Use short, declarative sentences for each step.
- Output any of the sentences in <Context> useful in your reasoning in the "Chain of Thought".
- Think step by step before writing the JSON. Your reasoning steps must appear in "Chain of Thought" as individual atomic facts.
"""


EXTENSION_RAG_PROMPT = \
"""<Context>
{context}
</Context>


<Question>
{question}
</Question>

Let’s answer the Question by reasoning step by step. 
"""

EXTENSION_VANILLA_PROMPT = \
"""<Question>
{question}
</Question>

Let’s answer the Question by reasoning step by step, and then parse your reasoning steps into atomic facts.
"""


EXTENSION_EXAMPLE_INSTRUCTION = \
"""Example of correct atomic reasoning:
Question: "Which cities border Lake Ontario?"
Output:
{
  "Answers": ["Toronto", "Hamilton"],
  "Chain of Thought": [
    "Lake Ontario borders several cities in Ontario, Canada.",
    "Toronto is located on the northwestern shore of Lake Ontario.",
    "Hamilton is located on the western shore of Lake Ontario."
  ]
}"""


WIKIDATA_REASONING_INSTRUCTION = \
"""You are a QA assistant skilled in answering questions about everything. In each input, you will be asked to answer a user query (QUESTION).
You will be asked to justify your answer thoroughly by providing your reasoning.
The input will be shaped as:

# QUERY:
QUESTION

Use your own information and reasoning to answer the question.
Please identify your thought process and final answers in a clear way. 
"""



MOVIES_REGULAR_REASONING_INSTRUCTION = \
"""You are a QA assistant skilled in answering questions about movies.
In each input, you might be provided with some external information (CONTEXT) including some SENTENCES, and a user query (QUESTION) about movies.
You may not need all of the context information to answer the query. Just look for the information that helps you find the answer to the QUESTION and connect them together if needed.
The input will be shaped as:

# CONTEXT: (optional)
A list of SENTENCES (optional)

# QUERY:
QUESTION

Answer the QUESTION using the SENTENCES in the CONTEXT if any, or use your own information and reasoning to answer the question.

Please identify your thought process and final answers in a clear way. 
"""



VANILLA_COT_INSTRUCTION = \
"""You are a QA assistant skilled in answering questions about movies.
You should answer a user query (QUESTION) about movies, and you will be asked to justify your answer by providing your chain of thought.
                    
# QUERY:
QUESTION


Answer the QUESTION using your knowledge about movies, and shape your output in a json format. 
Your json output should contain two keys: 

1. Include the key "Chain of Thought" valued as a list of sentences used in your thought process. Please include the exact form of the sentences in this list. 
2. Include the key "Answer" valued as a list of answers. Give your final answer in the shape of an array since the QUESTION might have multiple answers. 


For example, imagine a case where the question is: "What are the genre of movies directed by Director1?"
Imagine a case that Director1 has directed Movie1 and Movie2 with genres "Action" and "Comedy" respectively.
In this case, your answer should formatted as:
{
    "Chain of Thought": 
    [
        "Director1 has directed Movie1.", 
        "Movie1 is of genre Action.",
        "Director1 has directed Movie2.",
        "Movie2 is of genre Comedy."
    ],
    "Answers":
    [
        "Action", 
        "Comedy"
    ]
}
"""


KG_RAG_COT_INSTRUCTION = \
"""You are a QA assistant skilled in answering questions about movies.
In each input, you will be provided with some external information (CONTEXT) including some SENTENCES, and a user query (QUESTION) about movies. 
You don't need all of the context information to answer the query. Just look for the information that helps you find the answer to the QUESTION and connect them together if needed.
Answer the query only based on the CONTEXT you have been provided and demonstrate your thought process. 
                    
The input will be shaped as:
# CONTEXT:
A list of SENTENCES, separated by newlines

# QUERY:
QUESTION
######


Answer the QUESTION using the SENTENCES in the CONTEXT, and shape your output in a json format. 
Your json output should contain two keys: 

1. Include the key "Chain of Thought" valued as a list of SENTENCES used in your thought process. Please include the exact form of the SENTENCES in this list. 
2. Include the key "Answers" valued as a list of answers. Give your final answer in the shape of an array since the QUESTION might have multiple answers. 

For example, if you find Movie1, Movie2, and Movie3 as the answers to a query, the value of the key "Answers" should be: 
["Movie1", "Movie2", "Movie3"] 

Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.

"""


UMLS_SEED_ENTITY_INSTRUCTION = \
"""
You are interacting with a knowledge graph that contains definitions and relational information of medical terminologies.
A question will be asked about medical concepts, and you are expected to:

1. Extract Seed Entities: Based on the question, identify the seed entities that the search need to be started from them. There might be multiple seed entities.
2. Format the Output : Return in a structured JSON format with the key as "seed entities". For example:

{"seed entities": ["Term1", "Term2", ...]}

Please provide terms with title-case capitalization.
Do not answer the question. Just extract the key terminologies in the question. Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
"""


MOVIES_SEED_ENTITY_INSTRUCTION = \
"""You are interacting with a knowledge graph that contains information about movies with entities representing movies, their actors, directors, writers, release years, genres, and so on. 
A question will be asked about movies and their information, and you are expected to:

1. Extract Seed Entities: Based on the question, identify the seed entities that the search need to be started from them. There might be multiple seed entities.
2. Format the Output : Return in a structured JSON format with the key as "seed entities". For example:

{"seed entities": ["Actor1", "Actor2", ...]}

Do not answer the question. Just extract the seed entities in the question. Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
"""




WIKIDATA_SEED_ENTITY_INSTRUCTION = \
"""You are interacting with the WikiData general-purpose knowledge graph. 
A question will be asked about any subject, and you are expected to:

1. Extract Named Entities: 
    - Based on the question, identify the named entities that can be the seed entities for the knowledge graph search. 
    - Avoid general terms and look for informative entities since the knowledge graph is general and already contains lots of information. 
    - There might be multiple seed named entities, but they should all be in the question.
    - If you did not find any named entities, make sure you identify one key concept that the search can be started from.
2. Format the Output : Return in a structured JSON format with the key as "seed entities". For example:

{"seed entities": ["Entity1", "Entity2", ...]}

Just extract the seed entities directly in the question, and DO NOT ANSWER the question yourself. DO NOT return an empty list. 
Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
"""



SEED_ENTITY_PROMPT = \
"""# Question:
{}

In a JSON format, please provide the Named Entities of the question as instructed.
Do not answer the question, and only find the entities directly in the question.
"""


SEED_ENTITY_INSTRUCTIONS = {
    "wikidata": WIKIDATA_SEED_ENTITY_INSTRUCTION,
    "umls": UMLS_SEED_ENTITY_INSTRUCTION,
    "meta-qa": MOVIES_SEED_ENTITY_INSTRUCTION
}

# SEED_ENTITY_PROMPTS = {
#     "umls": UMLS_SEED_ENTITY_PROMPT,
#     "meta-qa": MOVIES_SEED_ENTITY_PROMPT
# }
