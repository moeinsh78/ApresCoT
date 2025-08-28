from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from typing import List, Set
from tqdm import tqdm
import pandas as pd
import numpy as np


ACCEPTABLE_SIMILARITY_TRESHOLD = 0.7

def match_nodes(nodes_set: Set[str], llm_final_answers: List[str]):
    answer_match_status = []
    node_to_answer_id = {}
    for i, answer in enumerate(llm_final_answers):
        if answer in nodes_set:
            node_to_answer_id[answer] = i + 1
            answer_match_status.append({"Answer": answer, "Index": i + 1})
        else:
            answer_match_status.append({"Answer": answer, "Index": "No Match"})

    return answer_match_status, node_to_answer_id


def match_edges(edge_descriptions: List[str], llm_cot: List[str]):
    encoded_edge_descriptions = get_sentence_encodings(edge_descriptions)
    
    matched_cot_list = []
    context_to_cot_match = {}
    for i, cot_step in enumerate(llm_cot):
        print("Reasoning step: ", cot_step)
        cot_step_embedding = get_sentence_encodings([cot_step])
        most_similar_context_sentence_id = get_most_similar_context_sentence_id(cot_step_embedding, encoded_edge_descriptions)
        
        print("Most Similar Sentence ID:", most_similar_context_sentence_id)
        if most_similar_context_sentence_id != -1:
            print("Most Similar Sentence:", edge_descriptions[most_similar_context_sentence_id - 1])
            print("\n\n")
            matched_cot_list.append({"COT Step": cot_step, "Most Similar Context ID": most_similar_context_sentence_id})
            context_to_cot_match[most_similar_context_sentence_id] = i + 1
        else:
            print("No Similar Sentence Found!\n\n")
            matched_cot_list.append({"COT Step": cot_step, "Most Similar Context ID": "No Match"})

    return matched_cot_list, context_to_cot_match



def get_sentence_encodings(sentence_list):
    """
    Get sentence embeddings using Sentence Transformer (MiniLM)
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence_list)

    return embeddings



def get_most_similar_context_sentence_id(cot_step_embedding, context_embeddings):
    """
    Get most similar context sentence ID for a given CoT step embedding by 
    calculating cosine similarity between CoT step and context sentences
    """
    similarity_scores = cosine_similarity(context_embeddings, cot_step_embedding).flatten()

    print("Similarity Scores:\n", similarity_scores)

    most_similar_context_sentence_id = np.argmax(similarity_scores) + 1
    if (similarity_scores[most_similar_context_sentence_id - 1] < ACCEPTABLE_SIMILARITY_TRESHOLD):
        return -1

    return most_similar_context_sentence_id



# def get_sentence_encodings(sentence_list):
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertModel.from_pretrained("distilbert-base-uncased")

#     context_sentences_df = pd.DataFrame(sentence_list, columns=['sentences'])
#     context_sentences_df["ids"] = context_sentences_df.index

#     batch_size = 1
#     embeddings = []

#     context_sentences_list = list(context_sentences_df.sentences)
#     encoded_context_sentences = tokenizer.batch_encode_plus(context_sentences_list, return_tensors='pt', max_length=512, padding=True, truncation=True)

#     for i in tqdm(range(0, len(encoded_context_sentences['input_ids']), batch_size)):
#         batch_input_ids = encoded_context_sentences['input_ids'][i:i + batch_size]
#         batch_attention_mask = encoded_context_sentences['attention_mask'][i:i + batch_size]
    
#         outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
    
#         batch_embeddings = outputs.last_hidden_state[:,0,:]
#         batch_embeddings = batch_embeddings.detach().numpy()

#         embeddings.append(batch_embeddings)

#     embeddings = np.concatenate(embeddings, axis=0)
    
#     return embeddings



# def get_most_similar_context_sentence_id(cot_step: str, context_embeddings):
#     tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
#     model = DistilBertModel.from_pretrained("distilbert-base-uncased")

#     encoded_cot_step = tokenizer(cot_step, return_tensors='pt', max_length=512, padding=True, truncation=True)

#     cot_step_vector_embedding = model(**encoded_cot_step).last_hidden_state[:, 0, :].detach().numpy()

#     # print("Context Embeddings Shape: ", context_embeddings.shape)
#     # print("COT Step Embeddings Shape: ", cot_step_vector_embedding.shape)
#     similarity_scores = cosine_similarity(context_embeddings, cot_step_vector_embedding).flatten()

#     print("Chain of Thought Step: ", cot_step)
#     print("Similarity Scores:\n", similarity_scores)

#     most_similar_context_sentence_id = np.argmax(similarity_scores) + 1
#     return most_similar_context_sentence_id

