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



def create_prompt(question: str, kg_name: str, rag: bool, llm: str, edge_descriptions: List[str]):
    if rag:
        return KG_RAG_COT_INSTRUCTION, build_rag_qa_prompt(edge_descriptions, question)
    else:
        return VANILLA_COT_INSTRUCTION, f"# QUERY:\n{question}\n\nLet's think step by step. Please provide any fact and sentence in your chain of thought separately. \n"




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
A question will be asked about medical concepts, and to provide a precise and relevant answer to this question, you are expected to:

1. Understand the Question Thoroughly: Analyze the question deeply to identify which specific medical terminologies and their interrelations, as extracted from the knowledge graph, are crucial for formulating an accurate response.

2. Extract Key Terminologies: Return at most 4 relevant medical terminologies based on their significance to the question.

3. Format the Output : Return in a structured JSON format with the key as "medical terminologies". For example:

{"medical terminologies": ["term1", "term2", ...]}

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



MOVIES_SEED_ENTITY_PROMPT = \
"""# Question:
{}

In a JSON format, please provide a list of seed entities that are crucial to start the knowledge graph search from them. 
"""


UMLS_SEED_ENTITY_PROMPT = \
"""# Question:
{}

In a JSON format, please provide the most relevant medical terminologies that are crucial for formulating an accurate response to the question. 
Please avoid including extra terminologies, and just consider the question.
"""


SEED_ENTITY_INSTRUCTIONS = {
    "umls": UMLS_SEED_ENTITY_INSTRUCTION,
    "meta-qa": MOVIES_SEED_ENTITY_INSTRUCTION
}

SEED_ENTITY_PROMPTS = {
    "umls": UMLS_SEED_ENTITY_PROMPT,
    "meta-qa": MOVIES_SEED_ENTITY_PROMPT
}


SEED_ENTITY_JSON_KEYS = {
    "umls": "medical terminologies",
    "meta-qa": "seed entities"
}
