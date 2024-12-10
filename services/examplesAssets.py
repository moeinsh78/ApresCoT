from typing import List, Dict

########################################################################################################################
# FORM SECTION
########################################################################################################################

USE_CASE_1_QUERY = "What were the release years of the films starred by Jean Rochefort?"
USE_CASE_1_LLM = "ChatGPT 3.5 + KG RAG"
USE_CASE_1_KG = "meta-qa"

USE_CASE_2_QUERY = "What were the release years of the films starred by Jean Rochefort?"
USE_CASE_2_LLM = "GPT-4o Mini"
USE_CASE_2_KG = "meta-qa"

USE_CASE_3_QUERY = "What types of animals are affected by dysfunctions caused by Fungus?"
USE_CASE_3_LLM = "ChatGPT 3.5 + KG RAG"
USE_CASE_3_KG = "umls"

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
    {'data': {'id': '1990', 'label': '[A2] 1990'}, 'classes': 'response'}, 
    {'data': {'id': 'yves robert', 'label': 'yves robert'}, 'classes': 'normal'}, 
    {'data': {'id': 'Francis Veber', 'label': 'Francis Veber'}, 'classes': 'normal'}, 
    {'data': {'id': 'Patrice Leconte', 'label': 'Patrice Leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'Claude Klotz', 'label': 'Claude Klotz'}, 'classes': 'normal'}, 
    {'data': {'id': 'hairdresser', 'label': 'hairdresser'}, 'classes': 'normal'}, 
    {'data': {'id': 'Bernard Blier', 'label': 'Bernard Blier'}, 'classes': 'normal'}, 
    {'data': {'id': 'English', 'label': 'English'}, 'classes': 'normal'}, 
    {'data': {'id': '1972', 'label': '[A1] 1972'}, 'classes': 'response'}, 
    {'data': {'id': 'Comedy', 'label': 'Comedy'}, 'classes': 'normal'}, 
    {'data': {'id': 'The Tall Blond Man with One Black Shoe', 'label': 'The Tall Blond Man with One Black Shoe'}, 'classes': 'normal'}, 
    {'data': {'id': 'Yves Robert', 'label': 'Yves Robert'}, 'classes': 'normal'}, 
    {'data': {'id': "The Hairdresser's Husband", 'label': "The Hairdresser's Husband"}, 'classes': 'normal'}, 
    {'data': {'source': 'Jean Rochefort', 'target': 'The Tall Blond Man with One Black Shoe', 'weight': '[S1] starred_actors', 'id': 'ed1ge1'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'Jean Rochefort', 'target': "The Hairdresser's Husband", 'weight': '[S3] starred_actors', 'id': 'ed1ge2'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'directed_by', 'id': 'ed1ge3'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'written_by', 'id': 'ed1ge4'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Francis Veber', 'weight': 'written_by', 'id': 'ed1ge5'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Bernard Blier', 'weight': 'starred_actors', 'id': 'ed1ge6'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Pierre Richard', 'weight': 'starred_actors', 'id': 'ed1ge7'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': '1972', 'weight': '[S2] release_year', 'id': 'ed1ge8'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'English', 'weight': 'in_language', 'id': 'ed1ge9'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'French', 'weight': 'in_language', 'id': 'ed1ge10'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Comedy', 'weight': 'has_genre', 'id': 'ed1ge11'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'yves robert', 'weight': 'has_tags', 'id': 'ed1ge12'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'pierre richard', 'weight': 'has_tags', 'id': 'ed1ge13'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'directed_by', 'id': 'ed1ge14'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'written_by', 'id': 'ed1ge15'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Claude Klotz', 'weight': 'written_by', 'id': 'ed1ge16'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Anna Galiena', 'weight': 'starred_actors', 'id': 'ed1ge17'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': '1990', 'weight': '[S4] release_year', 'id': 'ed1ge18'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'French', 'weight': 'in_language', 'id': 'ed1ge19'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'patrice leconte', 'weight': 'has_tags', 'id': 'ed1ge20'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'hairdresser', 'weight': 'has_tags', 'id': 'ed1ge21'}, 'classes': 'curved'}
]




####### TEMPORARY #######

USE_CASE_1_ALTER_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Jean Rochefort', 'label': 'Jean Rochefort'}, 'classes': 'source'}, 
    {'data': {'id': 'Pierre Richard', 'label': 'Pierre Richard'}, 'classes': 'normal'}, 
    {'data': {'id': 'French', 'label': 'French'}, 'classes': 'normal'}, 
    {'data': {'id': 'Anna Galiena', 'label': 'Anna Galiena'}, 'classes': 'normal'}, 
    {'data': {'id': 'patrice leconte', 'label': 'patrice leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'pierre richard', 'label': 'pierre richard'}, 'classes': 'normal'}, 
    {'data': {'id': '1990', 'label': '[A2] 1990'}, 'classes': 'response'}, 
    {'data': {'id': 'yves robert', 'label': 'yves robert'}, 'classes': 'normal'}, 
    {'data': {'id': 'Francis Veber', 'label': 'Francis Veber'}, 'classes': 'normal'}, 
    {'data': {'id': 'Patrice Leconte', 'label': 'Patrice Leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'Claude Klotz', 'label': 'Claude Klotz'}, 'classes': 'normal'}, 
    {'data': {'id': 'hairdresser', 'label': 'hairdresser'}, 'classes': 'normal'}, 
    {'data': {'id': 'Bernard Blier', 'label': 'Bernard Blier'}, 'classes': 'normal'}, 
    {'data': {'id': 'English', 'label': 'English'}, 'classes': 'normal'}, 
    {'data': {'id': '1972', 'label': '[A1] 1972'}, 'classes': 'response'}, 
    {'data': {'id': 'Comedy', 'label': 'Comedy'}, 'classes': 'normal'}, 
    {'data': {'id': 'The Tall Blond Man with One Black Shoe', 'label': 'The Tall Blond Man with One Black Shoe'}, 'classes': 'normal'}, 
    {'data': {'id': 'Yves Robert', 'label': 'Yves Robert'}, 'classes': 'normal'}, 
    {'data': {'id': "The Hairdresser's Husband", 'label': "The Hairdresser's Husband"}, 'classes': 'normal'}, 
    {'data': {'source': 'Jean Rochefort', 'target': 'The Tall Blond Man with One Black Shoe', 'weight': '[S1] starred_actors', 'id': 'ed12ge1'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'Jean Rochefort', 'target': "The Hairdresser's Husband", 'weight': '[S3] starred_actors', 'id': 'ed12ge2'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'directed_by', 'id': 'ed12ge3'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'written_by', 'id': 'ed12ge4'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Francis Veber', 'weight': 'written_by', 'id': 'ed12ge5'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Bernard Blier', 'weight': 'starred_actors', 'id': 'ed12ge6'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Pierre Richard', 'weight': 'starred_actors', 'id': 'ed12ge7'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': '1972', 'weight': '[S2] release_year', 'id': 'ed12ge8'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'English', 'weight': 'in_language', 'id': 'ed12ge9'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'French', 'weight': 'in_language', 'id': 'ed12ge10'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Comedy', 'weight': 'has_genre', 'id': 'ed12ge11'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'yves robert', 'weight': 'has_tags', 'id': 'ed12ge12'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'pierre richard', 'weight': 'has_tags', 'id': 'ed12ge13'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'directed_by', 'id': 'ed12ge14'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'written_by', 'id': 'ed12ge15'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Claude Klotz', 'weight': 'written_by', 'id': 'ed12ge16'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Anna Galiena', 'weight': 'starred_actors', 'id': 'ed12ge17'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': '1990', 'weight': '[S4] release_year', 'id': 'ed12ge18'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'French', 'weight': 'in_language', 'id': 'ed12ge19'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'patrice leconte', 'weight': 'has_tags', 'id': 'ed12ge20'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'hairdresser', 'weight': 'has_tags', 'id': 'ed12ge21'}, 'classes': 'curved'}
]




########################################################################################################################
# USE CASE 2 OUTPUTS
########################################################################################################################

USE_CASE_2_COT_STEPS: List[Dict] = [
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

USE_CASE_2_ANSWERS: List[Dict] = [
    {'Answer': '1995', 'Index': 'No Match'}, 
    {'Answer': '1972', 'Index': 2}, 
    {'Answer': '1990', 'Index': 3}, 
    {'Answer': '1967', 'Index': 'No Match'}, 
    {'Answer': '1974', 'Index': 'No Match'}
]

USE_CASE_2_EDGE_DESCS: List[str] = USE_CASE_1_EDGE_DESCS

USE_CASE_2_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Jean Rochefort', 'label': 'Jean Rochefort'}, 'classes': 'source'}, 
    {'data': {'id': 'Pierre Richard', 'label': 'Pierre Richard'}, 'classes': 'normal'}, 
    {'data': {'id': 'French', 'label': 'French'}, 'classes': 'normal'}, 
    {'data': {'id': 'Anna Galiena', 'label': 'Anna Galiena'}, 'classes': 'normal'}, 
    {'data': {'id': 'patrice leconte', 'label': 'patrice leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'pierre richard', 'label': 'pierre richard'}, 'classes': 'normal'}, 
    {'data': {'id': '1990', 'label': '[A3] 1990'}, 'classes': 'response'}, 
    {'data': {'id': 'yves robert', 'label': 'yves robert'}, 'classes': 'normal'}, 
    {'data': {'id': 'Francis Veber', 'label': 'Francis Veber'}, 'classes': 'normal'}, 
    {'data': {'id': 'Patrice Leconte', 'label': 'Patrice Leconte'}, 'classes': 'normal'}, 
    {'data': {'id': 'Claude Klotz', 'label': 'Claude Klotz'}, 'classes': 'normal'}, 
    {'data': {'id': 'hairdresser', 'label': 'hairdresser'}, 'classes': 'normal'}, 
    {'data': {'id': 'Bernard Blier', 'label': 'Bernard Blier'}, 'classes': 'normal'}, 
    {'data': {'id': 'English', 'label': 'English'}, 'classes': 'normal'}, 
    {'data': {'id': '1972', 'label': '[A2] 1972'}, 'classes': 'response'}, 
    {'data': {'id': 'Comedy', 'label': 'Comedy'}, 'classes': 'normal'}, 
    {'data': {'id': 'The Tall Blond Man with One Black Shoe', 'label': 'The Tall Blond Man with One Black Shoe'}, 'classes': 'normal'}, 
    {'data': {'id': 'Yves Robert', 'label': 'Yves Robert'}, 'classes': 'normal'}, 
    {'data': {'id': "The Hairdresser's Husband", 'label': "The Hairdresser's Husband"}, 'classes': 'normal'}, 
    {'data': {'source': 'Jean Rochefort', 'target': 'The Tall Blond Man with One Black Shoe', 'weight': '[S4] starred_actors', 'id': 'ed2ge1'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'Jean Rochefort', 'target': "The Hairdresser's Husband", 'weight': '[S6] starred_actors', 'id': 'ed2ge2'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'directed_by', 'id': 'ed2ge3'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Yves Robert', 'weight': 'written_by', 'id': 'ed2ge4'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Francis Veber', 'weight': 'written_by', 'id': 'ed2ge5'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Bernard Blier', 'weight': 'starred_actors', 'id': 'ed2ge6'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Pierre Richard', 'weight': 'starred_actors', 'id': 'ed2ge7'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': '1972', 'weight': '[S5] release_year', 'id': 'ed2ge8'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'English', 'weight': 'in_language', 'id': 'ed2ge9'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'French', 'weight': 'in_language', 'id': 'ed2ge10'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'Comedy', 'weight': 'has_genre', 'id': 'ed2ge11'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'yves robert', 'weight': 'has_tags', 'id': 'ed2ge12'}, 'classes': 'curved'}, 
    {'data': {'source': 'The Tall Blond Man with One Black Shoe', 'target': 'pierre richard', 'weight': 'has_tags', 'id': 'ed2ge13'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'directed_by', 'id': 'ed2ge14'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Patrice Leconte', 'weight': 'written_by', 'id': 'ed2ge15'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Claude Klotz', 'weight': 'written_by', 'id': 'ed2ge16'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'Anna Galiena', 'weight': 'starred_actors', 'id': 'ed2ge17'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': '1990', 'weight': '[S7] release_year', 'id': 'ed2ge18'}, 'classes': 'curved cot-edge'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'French', 'weight': 'in_language', 'id': 'ed2ge19'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'patrice leconte', 'weight': 'has_tags', 'id': 'ed2ge20'}, 'classes': 'curved'}, 
    {'data': {'source': "The Hairdresser's Husband", 'target': 'hairdresser', 'weight': 'has_tags', 'id': 'ed2ge21'}, 'classes': 'curved'}
]


########################################################################################################################
# USE CASE 3 OUTPUTS
########################################################################################################################

USE_CASE_3_COT_STEPS: List[Dict] = [
    {'COT Step': 'Fungus causes Mental or Behavioral Dysfunction.', 'Most Similar Context ID': 1},
    {'COT Step': 'Mental or Behavioral Dysfunction affects Mammal.', 'Most Similar Context ID': 4},
    {'COT Step': 'Mental or Behavioral Dysfunction affects Mental Process.', 'Most Similar Context ID': 5},
    {'COT Step': 'Mental or Behavioral Dysfunction affects Neoplastic Process.', 'Most Similar Context ID': 10},
    {'COT Step': 'Mental or Behavioral Dysfunction affects Reptile.', 'Most Similar Context ID': 24},
]

USE_CASE_3_ANSWERS: List[Dict] = [
    {'Answer': 'Mammal', 'Index': 1}, 
    {'Answer': 'Reptile', 'Index': 2}, 
    {'Answer': 'Bird', 'Index': 3}, 
]

USE_CASE_3_EDGE_DESCS: List[str] = [
    "Fungus causes Mental or Behavioral Dysfunction.",
    "Fungus is a Eukaryote.",
    "Fungus causes Experimental Model of Disease.",
    "Mental or Behavioral Dysfunction affects Mammal.",
    "Mental or Behavioral Dysfunction affects Mental Process.",
    "Experimental Model of Disease occurs in Injury or Poisoning.",
    "Fungus is the location of Enzyme.",
    "Fungus is the location of Hormone.",
    "Experimental Model of Disease occurs in Patient or Disabled Group.",
    "Mental or Behavioral Dysfunction affects Neoplastic Process.",
    "Fungus causes Cell or Molecular Dysfunction.",
    "Enzyme is a Chemical.",
    "Hormone disrupts Cell Component.",
    "Cell or Molecular Dysfunction affects Bird.",
    "Experimental Model of Disease occurs in Family Group.",
    "Experimental Model of Disease occurs in Professional or Occupational Group.",
    "Hormone causes Pathologic Function.",
    "Mental or Behavioral Dysfunction complicates Acquired Abnormality.",
    "Cell or Molecular Dysfunction affects Organism.",
    "Mental or Behavioral Dysfunction complicates Congenital Abnormality.",
    "Fungus interacts with Virus.",
    "Cell or Molecular Dysfunction affects Pathologic Function.",
    "Virus interacts with Bacterium.",
    "Mental or Behavioral Dysfunction affects Reptile.",
    "Cell or Molecular Dysfunction affects Mammal.",
    "Virus issue in Occupation or Discipline.",
    "Eukaryote is a Organism.",
    "Enzyme complicates Organism Function.",
    "Cell or Molecular Dysfunction affects Mental Process.",
    "Mental or Behavioral Dysfunction complicates Anatomical Abnormality.",
    "Mental or Behavioral Dysfunction complicates Cell or Molecular Dysfunction.",
    "Enzyme interacts with Vitamin.",
]

USE_CASE_3_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Vitamin', 'label': 'Vitamin'}, 'classes': 'normal'},
    {'data': {'id': 'Hormone', 'label': 'Hormone'}, 'classes': 'normal'},
    {'data': {'id': 'Organism', 'label': 'Organism'}, 'classes': 'normal'},
    {'data': {'id': 'Congenital Abnormality', 'label': 'Congenital Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Mental Process', 'label': 'Mental Process'}, 'classes': 'normal'},
    {'data': {'id': 'Fungus', 'label': 'Fungus'}, 'classes': 'source'},
    {'data': {'id': 'Bird', 'label': '[A3] Bird'}, 'classes': 'response'},
    {'data': {'id': 'Acquired Abnormality', 'label': 'Acquired Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Professional or Occupational Group', 'label': 'Professional or Occupational Group'}, 'classes': 'normal'},
    {'data': {'id': 'Enzyme', 'label': 'Enzyme'}, 'classes': 'normal'},
    {'data': {'id': 'Mammal', 'label': '[A1] Mammal'}, 'classes': 'response'},
    {'data': {'id': 'Eukaryote', 'label': 'Eukaryote'}, 'classes': 'normal'},
    {'data': {'id': 'Family Group', 'label': 'Family Group'}, 'classes': 'normal'},
    {'data': {'id': 'Anatomical Abnormality', 'label': 'Anatomical Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Pathologic Function', 'label': 'Pathologic Function'}, 'classes': 'normal'},
    {'data': {'id': 'Organism Function', 'label': 'Organism Function'}, 'classes': 'normal'},
    {'data': {'id': 'Cell or Molecular Dysfunction', 'label': 'Cell or Molecular Dysfunction'}, 'classes': 'normal'},
    {'data': {'id': 'Reptile', 'label': '[A2] Reptile'}, 'classes': 'response'},
    {'data': {'id': 'Neoplastic Process', 'label': 'Neoplastic Process'}, 'classes': 'normal'},
    {'data': {'id': 'Bacterium', 'label': 'Bacterium'}, 'classes': 'normal'},
    {'data': {'id': 'Injury or Poisoning', 'label': 'Injury or Poisoning'}, 'classes': 'normal'},
    {'data': {'id': 'Virus', 'label': 'Virus'}, 'classes': 'normal'},
    {'data': {'id': 'Patient or Disabled Group', 'label': 'Patient or Disabled Group'}, 'classes': 'normal'},
    {'data': {'id': 'Occupation or Discipline', 'label': 'Occupation or Discipline'}, 'classes': 'normal'},
    {'data': {'id': 'Chemical', 'label': 'Chemical'}, 'classes': 'normal'},
    {'data': {'id': 'Mental or Behavioral Dysfunction', 'label': 'Mental or Behavioral Dysfunction'}, 'classes': 'normal'},
    {'data': {'id': 'Experimental Model of Disease', 'label': 'Experimental Model of Disease'}, 'classes': 'normal'},
    {'data': {'id': 'Cell Component', 'label': 'Cell Component'}, 'classes': 'normal'},
    {'data': {'source': 'Fungus', 'target': 'Mental or Behavioral Dysfunction', 'weight': '[S1] causes', 'id': 'ed3ge1'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Fungus', 'target': 'Eukaryote', 'weight': 'isa', 'id': 'ed3ge2'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Experimental Model of Disease', 'weight': 'causes', 'id': 'ed3ge3'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Mammal', 'weight': '[S2] affects', 'id': 'ed3ge4'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Mental Process', 'weight': '[S3] affects', 'id': 'ed3ge5'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Injury or Poisoning', 'weight': 'occurs_in', 'id': 'ed3ge6'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Enzyme', 'weight': 'location_of', 'id': 'ed3ge7'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Hormone', 'weight': 'location_of', 'id': 'ed3ge8'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Patient or Disabled Group', 'weight': 'occurs_in', 'id': 'ed3ge9'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Neoplastic Process', 'weight': '[S4] affects', 'id': 'ed3ge10'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Fungus', 'target': 'Cell or Molecular Dysfunction', 'weight': 'causes', 'id': 'ed3ge11'}, 'classes': 'curved'},
    {'data': {'source': 'Enzyme', 'target': 'Chemical', 'weight': 'isa', 'id': 'ed3ge12'}, 'classes': 'curved'},
    {'data': {'source': 'Hormone', 'target': 'Cell Component', 'weight': 'disrupts', 'id': 'ed3ge13'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Bird', 'weight': 'affects', 'id': 'ed3ge14'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Family Group', 'weight': 'occurs_in', 'id': 'ed3ge15'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Professional or Occupational Group', 'weight': 'occurs_in', 'id': 'ed3ge16'}, 'classes': 'curved'},
    {'data': {'source': 'Hormone', 'target': 'Pathologic Function', 'weight': 'causes', 'id': 'ed3ge17'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Acquired Abnormality', 'weight': 'complicates', 'id': 'ed3ge18'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Organism', 'weight': 'affects', 'id': 'ed3ge19'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Congenital Abnormality', 'weight': 'complicates', 'id': 'ed3ge20'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Virus', 'weight': 'interacts_with', 'id': 'ed3ge21'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Pathologic Function', 'weight': 'affects', 'id': 'ed3ge22'}, 'classes': 'curved'},
    {'data': {'source': 'Virus', 'target': 'Bacterium', 'weight': 'interacts_with', 'id': 'ed3ge23'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Reptile', 'weight': '[S5] affects', 'id': 'ed3ge24'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Mammal', 'weight': 'affects', 'id': 'ed3ge25'}, 'classes': 'curved'},
    {'data': {'source': 'Virus', 'target': 'Occupation or Discipline', 'weight': 'issue_in', 'id': 'ed3ge26'}, 'classes': 'curved'},
    {'data': {'source': 'Eukaryote', 'target': 'Organism', 'weight': 'isa', 'id': 'ed3ge27'}, 'classes': 'curved'},
    {'data': {'source': 'Enzyme', 'target': 'Organism Function', 'weight': 'complicates', 'id': 'ed3ge28'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Mental Process', 'weight': 'affects', 'id': 'ed3ge29'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Anatomical Abnormality', 'weight': 'complicates', 'id': 'ed3ge30'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Cell or Molecular Dysfunction', 'weight': 'complicates', 'id': 'ed3ge31'}, 'classes': 'curved'},
    {'data': {'source': 'Enzyme', 'target': 'Vitamin', 'weight': 'interacts_with', 'id': 'ed3ge32'}, 'classes': 'curved'},
]


########################################################################################################################
# USE CASE 3 ALTERNATIVE OUTPUTS
########################################################################################################################

USE_CASE_3_ALTER_COT_STEPS: List[Dict] = [
    {'COT Step': 'Fungus causes Mental or Behavioral Dysfunction.', 'Most Similar Context ID': 1}, 
    {'COT Step': 'Mental or Behavioral Dysfunction affects Mammal.', 'Most Similar Context ID': 4}, 
    {'COT Step': 'Mental or Behavioral Dysfunction affects Reptile.', 'Most Similar Context ID': 24}, 
    {'COT Step': 'Fungus causes Cell or Molecular Dysfunction.', 'Most Similar Context ID': 11},
    {'COT Step': 'Cell or Molecular Dysfunction affects Bird.', 'Most Similar Context ID': 14},
    {'COT Step': 'Cell or Molecular Dysfunction affects Mammal.', 'Most Similar Context ID': 25},
]

USE_CASE_3_ALTER_ANSWERS: List[Dict] = [
    {'Answer': 'Mammal', 'Index': 1}, 
    {'Answer': 'Reptile', 'Index': 2}, 
    {'Answer': 'Bird', 'Index': 3}, 
]

USE_CASE_3_ALTER_EDGE_DESCS: List[str] = [
    "Fungus causes Mental or Behavioral Dysfunction.",
    "Fungus is a Eukaryote.",
    "Fungus causes Experimental Model of Disease.",
    "Mental or Behavioral Dysfunction affects Mammal.",
    "Mental or Behavioral Dysfunction affects Mental Process.",
    "Experimental Model of Disease occurs in Injury or Poisoning.",
    "Fungus is the location of Enzyme.",
    "Fungus is the location of Hormone.",
    "Experimental Model of Disease occurs in Patient or Disabled Group.",
    "Mental or Behavioral Dysfunction affects Neoplastic Process.",
    "Fungus causes Cell or Molecular Dysfunction.",
    "Enzyme is a Chemical.",
    "Hormone disrupts Cell Component.",
    "Cell or Molecular Dysfunction affects Bird.",
    "Experimental Model of Disease occurs in Family Group.",
    "Experimental Model of Disease occurs in Professional or Occupational Group.",
    "Hormone causes Pathologic Function.",
    "Mental or Behavioral Dysfunction complicates Acquired Abnormality.",
    "Cell or Molecular Dysfunction affects Organism.",
    "Mental or Behavioral Dysfunction complicates Congenital Abnormality.",
    "Fungus interacts with Virus.",
    "Cell or Molecular Dysfunction affects Pathologic Function.",
    "Virus interacts with Bacterium.",
    "Mental or Behavioral Dysfunction affects Reptile.",
    "Cell or Molecular Dysfunction affects Mammal.",
    "Virus issue in Occupation or Discipline.",
    "Eukaryote is a Organism.",
    "Enzyme complicates Organism Function.",
    "Cell or Molecular Dysfunction affects Mental Process.",
    "Mental or Behavioral Dysfunction complicates Anatomical Abnormality.",
    "Mental or Behavioral Dysfunction complicates Cell or Molecular Dysfunction.",
    "Enzyme interacts with Vitamin.",
]

USE_CASE_3_ALTER_GRAPH_ELEMENTS: List[Dict] = [
    {'data': {'id': 'Vitamin', 'label': 'Vitamin'}, 'classes': 'normal'},
    {'data': {'id': 'Hormone', 'label': 'Hormone'}, 'classes': 'normal'},
    {'data': {'id': 'Organism', 'label': 'Organism'}, 'classes': 'normal'},
    {'data': {'id': 'Congenital Abnormality', 'label': 'Congenital Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Mental Process', 'label': 'Mental Process'}, 'classes': 'normal'},
    {'data': {'id': 'Fungus', 'label': 'Fungus'}, 'classes': 'source'},
    {'data': {'id': 'Bird', 'label': '[A3] Bird'}, 'classes': 'response'},
    {'data': {'id': 'Acquired Abnormality', 'label': 'Acquired Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Professional or Occupational Group', 'label': 'Professional or Occupational Group'}, 'classes': 'normal'},
    {'data': {'id': 'Enzyme', 'label': 'Enzyme'}, 'classes': 'normal'},
    {'data': {'id': 'Mammal', 'label': '[A1] Mammal'}, 'classes': 'response'},
    {'data': {'id': 'Eukaryote', 'label': 'Eukaryote'}, 'classes': 'normal'},
    {'data': {'id': 'Family Group', 'label': 'Family Group'}, 'classes': 'normal'},
    {'data': {'id': 'Anatomical Abnormality', 'label': 'Anatomical Abnormality'}, 'classes': 'normal'},
    {'data': {'id': 'Pathologic Function', 'label': 'Pathologic Function'}, 'classes': 'normal'},
    {'data': {'id': 'Organism Function', 'label': 'Organism Function'}, 'classes': 'normal'},
    {'data': {'id': 'Cell or Molecular Dysfunction', 'label': 'Cell or Molecular Dysfunction'}, 'classes': 'normal'},
    {'data': {'id': 'Reptile', 'label': '[A2] Reptile'}, 'classes': 'response'},
    {'data': {'id': 'Neoplastic Process', 'label': 'Neoplastic Process'}, 'classes': 'normal'},
    {'data': {'id': 'Bacterium', 'label': 'Bacterium'}, 'classes': 'normal'},
    {'data': {'id': 'Injury or Poisoning', 'label': 'Injury or Poisoning'}, 'classes': 'normal'},
    {'data': {'id': 'Virus', 'label': 'Virus'}, 'classes': 'normal'},
    {'data': {'id': 'Patient or Disabled Group', 'label': 'Patient or Disabled Group'}, 'classes': 'normal'},
    {'data': {'id': 'Occupation or Discipline', 'label': 'Occupation or Discipline'}, 'classes': 'normal'},
    {'data': {'id': 'Chemical', 'label': 'Chemical'}, 'classes': 'normal'},
    {'data': {'id': 'Mental or Behavioral Dysfunction', 'label': 'Mental or Behavioral Dysfunction'}, 'classes': 'normal'},
    {'data': {'id': 'Experimental Model of Disease', 'label': 'Experimental Model of Disease'}, 'classes': 'normal'},
    {'data': {'id': 'Cell Component', 'label': 'Cell Component'}, 'classes': 'normal'},
    {'data': {'source': 'Fungus', 'target': 'Mental or Behavioral Dysfunction', 'weight': '[S1] causes', 'id': 'ed4ge1'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Fungus', 'target': 'Eukaryote', 'weight': 'isa', 'id': 'ed4ge2'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Experimental Model of Disease', 'weight': 'causes', 'id': 'ed4ge3'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Mammal', 'weight': '[S2] affects', 'id': 'ed4ge4'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Mental Process', 'weight': 'affects', 'id': 'ed4ge5'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Injury or Poisoning', 'weight': 'occurs_in', 'id': 'ed4ge6'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Enzyme', 'weight': 'location_of', 'id': 'ed4ge7'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Hormone', 'weight': 'location_of', 'id': 'ed4ge8'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Patient or Disabled Group', 'weight': 'occurs_in', 'id': 'ed4ge9'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Neoplastic Process', 'weight': 'affects', 'id': 'ed4ge10'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Cell or Molecular Dysfunction', 'weight': '[S4] causes', 'id': 'ed4ge11'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Enzyme', 'target': 'Chemical', 'weight': 'isa', 'id': 'ed4ge12'}, 'classes': 'curved'},
    {'data': {'source': 'Hormone', 'target': 'Cell Component', 'weight': 'disrupts', 'id': 'ed4ge13'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Bird', 'weight': '[S5] affects', 'id': 'ed4ge14'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Family Group', 'weight': 'occurs_in', 'id': 'ed4ge15'}, 'classes': 'curved'},
    {'data': {'source': 'Experimental Model of Disease', 'target': 'Professional or Occupational Group', 'weight': 'occurs_in', 'id': 'ed4ge16'}, 'classes': 'curved'},
    {'data': {'source': 'Hormone', 'target': 'Pathologic Function', 'weight': 'causes', 'id': 'ed4ge17'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Acquired Abnormality', 'weight': 'complicates', 'id': 'ed4ge18'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Organism', 'weight': 'affects', 'id': 'ed4ge19'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Congenital Abnormality', 'weight': 'complicates', 'id': 'ed4ge20'}, 'classes': 'curved'},
    {'data': {'source': 'Fungus', 'target': 'Virus', 'weight': 'interacts_with', 'id': 'ed4ge21'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Pathologic Function', 'weight': 'affects', 'id': 'ed4ge22'}, 'classes': 'curved'},
    {'data': {'source': 'Virus', 'target': 'Bacterium', 'weight': 'interacts_with', 'id': 'ed4ge23'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Reptile', 'weight': '[S3] affects', 'id': 'ed4ge24'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Mammal', 'weight': '[S6] affects', 'id': 'ed4ge25'}, 'classes': 'curved cot-edge'},
    {'data': {'source': 'Virus', 'target': 'Occupation or Discipline', 'weight': 'issue_in', 'id': 'ed4ge26'}, 'classes': 'curved'},
    {'data': {'source': 'Eukaryote', 'target': 'Organism', 'weight': 'isa', 'id': 'ed4ge27'}, 'classes': 'curved'},
    {'data': {'source': 'Enzyme', 'target': 'Organism Function', 'weight': 'complicates', 'id': 'ed4ge28'}, 'classes': 'curved'},
    {'data': {'source': 'Cell or Molecular Dysfunction', 'target': 'Mental Process', 'weight': 'affects', 'id': 'ed4ge29'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Anatomical Abnormality', 'weight': 'complicates', 'id': 'ed4ge30'}, 'classes': 'curved'},
    {'data': {'source': 'Mental or Behavioral Dysfunction', 'target': 'Cell or Molecular Dysfunction', 'weight': 'complicates', 'id': 'ed4ge31'}, 'classes': 'curved'},
    {'data': {'source': 'Enzyme', 'target': 'Vitamin', 'weight': 'interacts_with', 'id': 'ed4ge32'}, 'classes': 'curved'},
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

# USE_CASE_3_ALTER_OUTPUTS = {
#     "COT_STEPS": USE_CASE_3_ALTER_COT_STEPS, 
#     "ANSWERS": USE_CASE_3_ALTER_ANSWERS, 
#     "EDGE_DESCS": USE_CASE_3_ALTER_EDGE_DESCS, 
#     "GRAPH_ELEMENTS": USE_CASE_3_ALTER_GRAPH_ELEMENTS
# }

OUTPUT_LISTS = [USE_CASE_1_OUTPUTS, USE_CASE_2_OUTPUTS, USE_CASE_3_OUTPUTS]




def get_toy_form_for_use_case(use_case_num: int):
    return TOY_QUERIES[use_case_num-1], TOY_LLMS[use_case_num-1], TOY_KGS[use_case_num-1]
