from typing import List, Dict, Sequence
import re


def build_rag_qa_prompt(edge_descriptions: List[str], question: str):
    question = re.sub(r'[\[\]]', '\"', question)
    edge_description_str = ""
    for i, desc in enumerate(edge_descriptions):
        edge_description_str += (desc + "\n")

    prompt = """# CONTEXT:
{}

# QUERY:
{}

Think step by step, and write down all of the sentences that you used in your thought process to answer the question.
""".format(edge_description_str, question)
    return prompt

def build_regular_rag_qa_prompt(edge_descriptions: List[str], question: str):
    question = re.sub(r'[\[\]]', '\"', question)
    edge_description_str = ""
    for i, desc in enumerate(edge_descriptions):
        edge_description_str += (desc + "\n")

    prompt = """# CONTEXT:
{}

# QUERY:
{}

Please provide your answer to this question based on the provided context, and explain your reasoning step by step by pointing out the information you used.
""".format(edge_description_str, question)
    return prompt



def create_prompt(question: str, kg_name: str, rag: bool, llm: str, edge_descriptions: List[str], new_reasoning: bool = True):
    if llm == "o3-mini":
        return "", f"{question}\nAnswer this question by thinking step by step, and provide your reasoning process.\n"
    if new_reasoning:
        if kg_name == "wikidata":
            return WIKIDATA_REASONING_INSTRUCTION, f"# QUERY:\n{question}\n\nPlease provide your answer to this question and explain your reasoning step by step.\n"
        if rag:
            return MOVIES_REGULAR_REASONING_INSTRUCTION, build_regular_rag_qa_prompt(edge_descriptions, question) 
        else:
            return MOVIES_REGULAR_REASONING_INSTRUCTION, f"# QUERY:\n{question}\n\nPlease provide your answer to this question and explain your reasoning step by step.\n"
    if rag:
        return KG_RAG_COT_INSTRUCTION, build_rag_qa_prompt(edge_descriptions, question)
    else:
        return VANILLA_COT_INSTRUCTION, f"# QUERY:\n{question}\n\nLet's think step by step. Please provide your reasoning steps and answers clearly. \n"


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
    "Answer":
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
2. Include the key "Answer" valued as a list of answers. Give your final answer in the shape of an array since the QUESTION might have multiple answers. 

For example, if you find Movie1, Movie2, and Movie3 as the answers to a query, the value of the key "Answer" should be: 
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
"""
You are interacting with a knowledge graph that contains information about movies with entities representing movies, their actors, directors, writers, release years, genres, and so on. 
A question will be asked about movies and their information, and you are expected to:

1. Extract Seed Entities: Based on the question, identify the seed entities that the search need to be started from them. There might be multiple seed entities.
2. Format the Output : Return in a structured JSON format with the key as "seed entities". For example:

{"seed entities": ["Actor1", "Actor2", ...]}

Do not answer the question. Just extract the seed entities in the question. Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
"""




WIKIDATA_SEED_ENTITY_INSTRUCTION = \
"""
You are interacting with the WikiData general-purpose knowledge graph. 
A question will be asked about any subject, and you are expected to:

1. Extract Seed Entities: Based on the question, identify the key concepts that can be the seed entities for the knowledge graph search. Avoid general terms and look for informative entities since the knowledge graph is general and already contains lots of information. There might be multiple seed entities, but they should all be in the question.
2. Format the Output : Return in a structured JSON format with the key as "seed entities". For example:

{"seed entities": ["Entity1", "Entity2", ...]}

Just extract the seed entities directly in the question, and DO NOT ANSWER the question yourself.
Failure to do so could result in incorrect information being provided to users, which could lead to a loss of trust in our service.
"""



SEED_ENTITY_PROMPT = \
"""# Question:
{}

In a JSON format, please provide a list of seed entities that are crucial to start the knowledge graph search from them to answer the question above.
Do not answer the question, and only find the entities directly in the question.
"""


# SEED_ENTITY_PROMPT = \
# """# Question:
# {}

# In a JSON format, please provide the most relevant medical terminologies that are crucial for formulating an accurate response to the question. 
# Please avoid including extra terminologies, and just consider the question.
# """


SEED_ENTITY_INSTRUCTIONS = {
    "wikidata": WIKIDATA_SEED_ENTITY_INSTRUCTION,
    "umls": UMLS_SEED_ENTITY_INSTRUCTION,
    "meta-qa": MOVIES_SEED_ENTITY_INSTRUCTION
}

# SEED_ENTITY_PROMPTS = {
#     "umls": UMLS_SEED_ENTITY_PROMPT,
#     "meta-qa": MOVIES_SEED_ENTITY_PROMPT
# }
