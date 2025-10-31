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



def create_prompt(question: str, rag: bool, edge_descriptions: List[str]):
    if rag:
        return EXTENSION_RAG_FREE_FORM_INSTRUCTION, EXTENSION_RAG_PROMPT.format(context="\n".join(edge_descriptions), question=question)
    else:
        return EXTENSION_FREE_FORM_INSTRUCTION, EXTENSION_VANILLA_PROMPT.format(question=question)






EXTENSION_FREE_FORM_INSTRUCTION = \
"""You are a reasoning assistant for question answering.

For each question, follow these steps:

1. Think step by step in natural language and output your reasoning and every bit of information you might use. 
    Prefer short, complete sentences, and avoid bullets or numbering.

2. After you finish the reasoning, output a final section that begins exactly with:
   ---FINAL ANSWERS---
   Then list the final answers, one per line.
   - Do not include bullets or numbering.
   - Deduplicate answers and remove trailing punctuation.

Output Format:

<your reasoning here with complete sentences>

---FINAL ANSWERS---
<one answer per line>
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

##############################################################################
############################# BEING USED NOW #################################

# EXTENSION_FREE_FORM_INSTRUCTION = \
# """You are a reasoning assistant for question answering.

# For each question, follow these steps:

# 1. Think step by step, showing your reasoning in natural language in the REASONING section.

# 2. After reasoning, rewrite your reasoning steps by extracting all the atomic factual statement you used or implied.
#    - list the sentences in the PARSED ATOMIC REASONING STEPS section.
#    - Each atomic fact must express exactly one piece of information.
#    - Avoid combining facts with words like "and", "also", or "which", or separating them with commas.

# 3. Then extract from your reasoning all the final answers you found by listing them in the FINAL ANSWERS section.

# 4. Use the following exact format:

# ---REASONING---
# <your detailed reasoning here>

# ---PARSED ATOMIC REASONING STEPS---
# <each piece of information on a new line>

# ---FINAL ANSWERS---
# <list of final answers each in a separate line>

# Do not include anything outside these three sections.
# """


# EXTENSION_RAG_FREE_FORM_INSTRUCTION = \
# """You are a reasoning assistant for question answering.
# You will be given a question and a block of context information retrieved from a knowledge base.

# For each question, follow these steps:

# 1. Read the Context carefully and write down your reasoning step by step in the REASONING section.

# 2. Rewrite your reasoning steps by extracting all the atomic factual statements in the Context that you used or implied.
#    - List these in the PARSED ATOMIC REASONING STEPS section.
#    - Each atomic fact must express exactly one piece of information.
#    - Avoid combining facts with words like "and", "also", or "which", or separating them with commas.

# 3. Then extract from your reasoning all final answers you found and list them in the
#    FINAL ANSWERS section (one per line). 

# 4. Use the following exact format:

# ---REASONING---
# <your detailed reasoning here>

# ---PARSED ATOMIC REASONING STEPS---
# <each statement from the context on a new line>

# ---FINAL ANSWERS---
# <list of final answers each in a separate line>

# Do not include anything outside these three sections.
# """

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




##############################################################################
################### SEED ENTITY INSTRUCTION AND PROMPTS ######################



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
