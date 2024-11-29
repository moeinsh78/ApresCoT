from typing import List, Dict

########################################################################################################################
# FORM SECTION
########################################################################################################################

USE_CASE_1_QUERY = "What were the release years of the films starred by Jean Rochefort?"
USE_CASE_1_LLM = "ChatGPT 3.5 + KG RAG"
USE_CASE_1_KG = "MetaQA Movies"

USE_CASE_2_QUERY = "Medical Question Placeholder"
USE_CASE_2_LLM = "UC LLM Placeholder"
USE_CASE_2_KG = "UMLS Relations"

USE_CASE_3_QUERY = "What were the release years of the films starred by Jean Rochefort?"
USE_CASE_3_LLM = "GPT-4o Mini"
USE_CASE_3_KG = "MetaQA Movies"

TOY_QUERIES = [USE_CASE_1_QUERY, USE_CASE_2_QUERY, USE_CASE_3_QUERY]
TOY_LLMS = [USE_CASE_1_LLM, USE_CASE_2_LLM, USE_CASE_3_LLM]
TOY_KGS = [USE_CASE_1_KG, USE_CASE_2_KG, USE_CASE_3_KG]


########################################################################################################################
# USE CASE 1 OUTPUTS
########################################################################################################################

USE_CASE_1_COT_STEPS: List[Dict] = [
    {"COT Step": "Actor \"Jean Rochefort\" starred in \"The Tall Blond Man with One Black Shoe\".", "Most Similar Context ID": 1},
    {"COT Step": "Movie \"The Tall Blond Man with One Black Shoe\" was released in 1972.", "Most Similar Context ID": 8},
    {"COT Step": "Actor \"Jean Rochefort\" starred in \"The Hairdresser\'s Husband\".", "Most Similar Context ID": 2},
    {"COT Step": "Movie \"The Hairdresser's Husband\" was released in 1990.", "Most Similar Context ID": 18}
]

USE_CASE_1_ANSWERS: List[Dict] = [
    {"Answer": "1972", "Index": 1},
    {"Answer": "1990", "Index": 2},
]


USE_CASE_1_EDGE_DESCS: List[str] = [
    'Actor "Jean Rochefort" starred in "The Tall Blond Man with One Black Shoe".', 
    'Actor "Jean Rochefort" starred in "The Hairdresser\'s Husband".', 
    'Movie "The Tall Blond Man with One Black Shoe" was directed by "Yves Robert".', 
    'Movie "The Tall Blond Man with One Black Shoe" was written by "Yves Robert".', 
    'Movie "The Tall Blond Man with One Black Shoe" was written by "Francis Veber".', 
    'Actor "Bernard Blier" starred in "The Tall Blond Man with One Black Shoe".', 
    'Actor "Pierre Richard" starred in "The Tall Blond Man with One Black Shoe".', 
    'Movie "The Tall Blond Man with One Black Shoe" was released in 1972.', 
    'Movie "The Tall Blond Man with One Black Shoe" is in English language.', 
    'Movie "The Tall Blond Man with One Black Shoe" is in French language.', 
    'Movie "The Tall Blond Man with One Black Shoe" has genre Comedy.', 
    'Movie "The Tall Blond Man with One Black Shoe" is described with "yves robert" tag.', 
    'Movie "The Tall Blond Man with One Black Shoe" is described with "pierre richard" tag.', 
    'Movie "The Hairdresser\'s Husband" was directed by "Patrice Leconte".', 
    'Movie "The Hairdresser\'s Husband" was written by "Patrice Leconte".', 
    'Movie "The Hairdresser\'s Husband" was written by "Claude Klotz".', 
    'Actor "Anna Galiena" starred in "The Hairdresser\'s Husband".', 
    'Movie "The Hairdresser\'s Husband" was released in 1990.', 
    'Movie "The Hairdresser\'s Husband" is in French language.', 
    'Movie "The Hairdresser\'s Husband" is described with "patrice leconte" tag.', 
    'Movie "The Hairdresser\'s Husband" is described with "hairdresser" tag.'
]


USE_CASE_1_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Jean Rochefort', 'label': 'Jean Rochefort'}, 'classes': 'source'}, 
    {'data': {'id': 'Pierre Richard', 'label': 'Pierre Richard'}, 'classes': 'normal'}, 
    {'data': {'id': 'French', 'label': 'French'}, 'classes': 'normal'}, 
    {'data': {'id': 'Anna Galiena', 'label': 'Anna Galiena'}, 'classes': 'normal'}, 
    {'data': {'id': 'patrice leconte', 'label': 'patrice leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'pierre richard', 'label': 'pierre richard'}, 'classes': 'normal'}, 
    {'data': {'id': '1990', 'label': '[2] 1990'}, 'classes': 'response'}, 
    {'data': {'id': 'yves robert', 'label': 'yves robert'}, 'classes': 'normal'}, 
    {'data': {'id': 'Francis Veber', 'label': 'Francis Veber'}, 'classes': 'normal'}, 
    {'data': {'id': 'Patrice Leconte', 'label': 'Patrice Leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'Claude Klotz', 'label': 'Claude Klotz'}, 'classes': 'normal'}, 
    {'data': {'id': 'hairdresser', 'label': 'hairdresser'}, 'classes': 'normal'}, 
    {'data': {'id': 'Bernard Blier', 'label': 'Bernard Blier'}, 'classes': 'normal'}, 
    {'data': {'id': 'English', 'label': 'English'}, 'classes': 'normal'}, 
    {'data': {'id': '1972', 'label': '[1] 1972'}, 'classes': 'response'}, 
    {'data': {'id': 'Comedy', 'label': 'Comedy'}, 'classes': 'normal'}, 
    {'data': {'id': 'The Tall Blond Man with One Black Shoe', 'label': 'The Tall Blond Man with One Black Shoe'}, 'classes': 'normal'}, 
    {'data': {'id': 'Yves Robert', 'label': 'Yves Robert'}, 'classes': 'normal'}, 
    {'data': {'id': "The Hairdresser's Husband", 'label': "The Hairdresser's Husband"}, 'classes': 'normal'}, 
    {'data': {'source': 'Jean Rochefort', 'target': 'The Tall Blond Man with One Black Shoe', 'weight': '[1] starred_actors', 'id': 'edge1'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'Jean Rochefort', 'target': "The Hairdresser's Husband", 'weight': '[3] starred_actors', 'id': 'edge2'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'directed_by', 'id': 'edge3'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'written_by', 'id': 'edge4'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Francis Veber', 'weight': 'written_by', 'id': 'edge5'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Bernard Blier', 'weight': 'starred_actors', 'id': 'edge6'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Pierre Richard', 'weight': 'starred_actors', 'id': 'edge7'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': '1972', 'weight': '[2] release_year', 'id': 'edge8'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'English', 'weight': 'in_language', 'id': 'edge9'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'French', 'weight': 'in_language', 'id': 'edge10'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Comedy', 'weight': 'has_genre', 'id': 'edge11'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'yves robert', 'weight': 'has_tags', 'id': 'edge12'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'pierre richard', 'weight': 'has_tags', 'id': 'edge13'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'directed_by', 'id': 'edge14'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'written_by', 'id': 'edge15'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Claude Klotz', 'weight': 'written_by', 'id': 'edge16'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Anna Galiena', 'weight': 'starred_actors', 'id': 'edge17'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': '1990', 'weight': '[4] release_year', 'id': 'edge18'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'French', 'weight': 'in_language', 'id': 'edge19'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'patrice leconte', 'weight': 'has_tags', 'id': 'edge20'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'hairdresser', 'weight': 'has_tags', 'id': 'edge21'}, 'classes': 'curved'}
]

########################################################################################################################
# USE CASE 2 OUTPUTS
########################################################################################################################

USE_CASE_2_COT_STEPS: List[Dict] = []

USE_CASE_2_ANSWERS: List[Dict] = []

USE_CASE_2_EDGE_DESCS: List[str] = []

USE_CASE_2_GRAPH_ELEMENTS: List[Dict] = []

########################################################################################################################
# USE CASE 3 OUTPUTS
########################################################################################################################

USE_CASE_3_COT_STEPS: List[Dict] = [
    {'COT Step': 'Jean Rochefort was a prominent French actor known for his work in various films.', 'Most Similar Context ID': 'No Match'},
    {'COT Step': "Jean Rochefort starred in the film 'The Horseman on the Roof'.", 'Most Similar Context ID': 'No Match'},
    {'COT Step': " The Horseman on the Roof' was released in 1995.", 'Most Similar Context ID': 'No Match'},
    {'COT Step': "He also starred in 'The Tall Blond Man with One Black Shoe'.", 'Most Similar Context ID': 1},
    {'COT Step': "'The Tall Blond Man with One Black Shoe' was released in 1972.", 'Most Similar Context ID': 8},
    {'COT Step': "Another notable film he starred in is 'The Hairdresser's Husband'.", 'Most Similar Context ID': 2},
    {'COT Step': "'The Hairdresser's Husband' was released in 1990.", 'Most Similar Context ID': 18},
    {'COT Step': "Rochefort appeared in 'The Last Adventure'.", 'Most Similar Context ID': 'No Match'},
    {'COT Step': "Movie 'The Last Adventure' which was released in 1967.", 'Most Similar Context ID': 'No Match'},
    {'COT Step': "He was also in 'The Return of the Tall Blond Man' released in 1974.", 'Most Similar Context ID': 'No Match'}    
]

USE_CASE_3_ANSWERS: List[Dict] = [
    {'Answer': '1995', 'Index': 'No Match'}, 
    {'Answer': '1972', 'Index': 2}, 
    {'Answer': '1990', 'Index': 3}, 
    {'Answer': '1967', 'Index': 'No Match'}, 
    {'Answer': '1974', 'Index': 'No Match'}
]

USE_CASE_3_EDGE_DESCS: List[str] = USE_CASE_1_EDGE_DESCS

USE_CASE_3_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Jean Rochefort', 'label': 'Jean Rochefort'}, 'classes': 'source'}, 
    {'data': {'id': 'Pierre Richard', 'label': 'Pierre Richard'}, 'classes': 'normal'}, 
    {'data': {'id': 'French', 'label': 'French'}, 'classes': 'normal'}, 
    {'data': {'id': 'Anna Galiena', 'label': 'Anna Galiena'}, 'classes': 'normal'}, 
    {'data': {'id': 'patrice leconte', 'label': 'patrice leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'pierre richard', 'label': 'pierre richard'}, 'classes': 'normal'}, 
    {'data': {'id': '1990', 'label': '[3] 1990'}, 'classes': 'response'}, 
    {'data': {'id': 'yves robert', 'label': 'yves robert'}, 'classes': 'normal'}, 
    {'data': {'id': 'Francis Veber', 'label': 'Francis Veber'}, 'classes': 'normal'}, 
    {'data': {'id': 'Patrice Leconte', 'label': 'Patrice Leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'Claude Klotz', 'label': 'Claude Klotz'}, 'classes': 'normal'}, 
    {'data': {'id': 'hairdresser', 'label': 'hairdresser'}, 'classes': 'normal'}, 
    {'data': {'id': 'Bernard Blier', 'label': 'Bernard Blier'}, 'classes': 'normal'}, 
    {'data': {'id': 'English', 'label': 'English'}, 'classes': 'normal'}, 
    {'data': {'id': '1972', 'label': '[2] 1972'}, 'classes': 'response'}, 
    {'data': {'id': 'Comedy', 'label': 'Comedy'}, 'classes': 'normal'}, 
    {'data': {'id': 'The Tall Blond Man with One Black Shoe', 'label': 'The Tall Blond Man with One Black Shoe'}, 'classes': 'normal'}, 
    {'data': {'id': 'Yves Robert', 'label': 'Yves Robert'}, 'classes': 'normal'}, 
    {'data': {'id': "The Hairdresser's Husband", 'label': "The Hairdresser's Husband"}, 'classes': 'normal'}, 
    {'data': {'source': 'Jean Rochefort', 'target': 'The Tall Blond Man with One Black Shoe', 'weight': '[3] starred_actors', 'id': 'edge1'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'Jean Rochefort', 'target': "The Hairdresser's Husband", 'weight': '[5] starred_actors', 'id': 'edge2'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'directed_by', 'id': 'edge3'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'written_by', 'id': 'edge4'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Francis Veber', 'weight': 'written_by', 'id': 'edge5'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Bernard Blier', 'weight': 'starred_actors', 'id': 'edge6'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Pierre Richard', 'weight': 'starred_actors', 'id': 'edge7'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': '1972', 'weight': '[4] release_year', 'id': 'edge8'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'English', 'weight': 'in_language', 'id': 'edge9'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'French', 'weight': 'in_language', 'id': 'edge10'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Comedy', 'weight': 'has_genre', 'id': 'edge11'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'yves robert', 'weight': 'has_tags', 'id': 'edge12'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'pierre richard', 'weight': 'has_tags', 'id': 'edge13'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'directed_by', 'id': 'edge14'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'written_by', 'id': 'edge15'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Claude Klotz', 'weight': 'written_by', 'id': 'edge16'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Anna Galiena', 'weight': 'starred_actors', 'id': 'edge17'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': '1990', 'weight': '[6] release_year', 'id': 'edge18'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'French', 'weight': 'in_language', 'id': 'edge19'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'patrice leconte', 'weight': 'has_tags', 'id': 'edge20'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'hairdresser', 'weight': 'has_tags', 'id': 'edge21'}, 'classes': 'curved'}
]


########################################################################################################################
# OUTPUTS
########################################################################################################################

USE_CASE_1_OUTPUTS = {
    "COT_STEPS": USE_CASE_1_COT_STEPS, 
    "ANSWERS": USE_CASE_1_ANSWERS, 
    "EDGE_DESCS": USE_CASE_1_EDGE_DESCS, 
    "GRAPH_ELEMENTS": USE_CASE_1_GRAPH_ELEMENTS
}

USE_CASE_2_OUTPUTS = {
    "COT_STEPS": USE_CASE_2_COT_STEPS, 
    "ANSWERS": USE_CASE_2_ANSWERS, 
    "EDGE_DESCS": USE_CASE_2_EDGE_DESCS, 
    "GRAPH_ELEMENTS": USE_CASE_2_GRAPH_ELEMENTS
}

USE_CASE_3_OUTPUTS = {
    "COT_STEPS": USE_CASE_3_COT_STEPS, 
    "ANSWERS": USE_CASE_3_ANSWERS, 
    "EDGE_DESCS": USE_CASE_3_EDGE_DESCS, 
    "GRAPH_ELEMENTS": USE_CASE_3_GRAPH_ELEMENTS
}


OUTPUT_LISTS = [USE_CASE_1_OUTPUTS, USE_CASE_2_OUTPUTS, USE_CASE_3_OUTPUTS]




def get_toy_form_for_use_case(use_case_num: int):
    return TOY_QUERIES[use_case_num-1], TOY_LLMS[use_case_num-1], TOY_KGS[use_case_num-1]


# def get_toy_insights_for_use_case(use_case_num: int):
#     desired_use_case_outputs = OUTPUT_LISTS[use_case_num-1]
#     doc_section = build_edge_description_table(desired_use_case_outputs["EDGE_DESCS"])
#     subgraph_section = draw_subgraph(desired_use_case_outputs["GRAPH_ELEMENTS"], SUBGRAPH_FIGURE_ID)
        
#     QA_section = None
#     llm_answers_section = build_llm_answers_table(desired_use_case_outputs["ANSWERS"])
#     llm_cot_section = build_llm_cot_table(desired_use_case_outputs["COT_STEPS"])

#     return doc_section, subgraph_section, QA_section, llm_answers_section, llm_cot_section 