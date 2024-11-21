import dash
from typing import List, Optional
from dash import Input, Output, State, callback, ctx
from services.components import (
    build_kage_page_layout,
    build_edge_description_table, 
    build_qa_info_table, 
    build_llm_answers_table,
    build_llm_cot_table,
    draw_subgraph
)
from services.config import APP_NAME, LLM_NAMES, SYSTEM_NAMES

from rage.qa import (
    perform_umls_qa, 
    perform_movies_qa, 
    get_used_sentences_ids, 
    build_cyto_subgraph_elements_list, 
    retrieve_subgraph,
    perform_vanilla_qa,
    remove_element
)

dash.register_page(__name__, title=APP_NAME)



QUESTION_INPUT_ID = "KAGE_PAGE_QUESTION_INPUT_ID"
HOP_SELECT_ID = "KAGE_PAGE_HOP_SELECT_ID"
LLM_SELECT_ID = "KAGE_PAGE_LLM_SELECT_ID"
KG_SELECT_ID = "KAGE_PAGE_KG_SELECT_ID"
GENERATE_BUTTON_ID = "KAGE_PAGE_GENERATE_BUTTON_ID"
DESCRIPTION_TABLE_CONTAINER_ID = "KAGE_PAGE_DESCRIPTION_TABLE_CONTAINER_ID"
SUBGRAPH_CONTAINER_ID = "KAGE_PAGE_SUBGRAPH_CONTAINER_ID"
SUBGRAPH_FIGURE_ID = "KAGE_PAGE_SUBGRAPH_FIGURE_ID"
QA_INFO_TABLE_CONTAINER_ID = "KAGE_PAGE_INFO_TABLE_CONTAINER_ID"
LLM_ANSWERS_TABLE_CONTAINER_ID = "KAGE_PAGE_LLM_ANSWERS_TABLE_CONTAINER_ID"
LLM_COT_TABLE_CONTAINER_ID = "KAGE_PAGE_LLM_COT_TABLE_CONTAINER_ID"
RESULT_SECTION_ID = "KAGE_RESULT_SECTION_ID"


layout = build_kage_page_layout(
    QUESTION_INPUT_ID, HOP_SELECT_ID, LLM_SELECT_ID, KG_SELECT_ID,
    DESCRIPTION_TABLE_CONTAINER_ID, SUBGRAPH_CONTAINER_ID,
    QA_INFO_TABLE_CONTAINER_ID, LLM_ANSWERS_TABLE_CONTAINER_ID, 
    LLM_COT_TABLE_CONTAINER_ID, GENERATE_BUTTON_ID, RESULT_SECTION_ID
)



@callback(
    Output(DESCRIPTION_TABLE_CONTAINER_ID, "children"),
    Output(SUBGRAPH_CONTAINER_ID, "children"),
    Output(QA_INFO_TABLE_CONTAINER_ID, "children"),
    Output(LLM_ANSWERS_TABLE_CONTAINER_ID, "children"),
    Output(LLM_COT_TABLE_CONTAINER_ID, "children"),
    Input(GENERATE_BUTTON_ID, 'n_clicks'),    
    State(QUESTION_INPUT_ID, "value"),
    # State(HOP_SELECT_ID, "value"),
    State(LLM_SELECT_ID, "value"),
    State(KG_SELECT_ID, "value"),
)
def on_generate(
    gen_btn_n_clicks: int, question: str, # hops_count: int, 
    supported_qa_model: str, kg_name: str, 
):
    input_is_invalid = (
        question is None
        or kg_name is None
        or len(question) == 0
        or len(kg_name) == 0
    )

    doc_section, subgraph_section = None, None
    QA_section, llm_answers_section, llm_cot_section = None, None, None

    hops_count = 2

    if not input_is_invalid:
        system = SYSTEM_NAMES[supported_qa_model]
        llm = LLM_NAMES[supported_qa_model]
        print("REQUESTED LLM:", llm)
        print("REQUESTED system:", system)
        
        if system == "vanilla-gpt-3.5" or system == "vanilla-gpt-4o-mini":
            seed_nodes, edge_dict_list, edge_description_list = retrieve_subgraph(question, dataset = "MOVIES", depth = hops_count)
            instruction_msg, prompt, llm_response, cot_response_list, final_answer_list = \
                perform_vanilla_qa(llm, question)
        
            matched_cot_list, context_to_cot_match = get_used_sentences_ids(edge_description_list, cot_response_list)
            subgraph_elements_list = []

            subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, edge_dict_list, context_to_cot_match, final_answer_list)
            
            # subgraph_elements_list = build_cyto_subgraph_elements_list(edge_dict_list, [], response_list)

            doc_section = build_edge_description_table(edge_description_list)
            subgraph_section = draw_subgraph(subgraph_elements_list, SUBGRAPH_FIGURE_ID)
        
            QA_section = build_qa_info_table(instruction_msg, prompt, llm_response)
            llm_answers_section = build_llm_answers_table(final_answer_list)
            llm_cot_section = build_llm_cot_table(matched_cot_list)


        elif system == "kg-gpt-3.5":
            seed_nodes, edge_dict_list, edge_description_list = retrieve_subgraph(question, dataset = "MOVIES", depth = hops_count)
            instruction_msg, prompt, llm_response, cot_response_list, final_answer_list = \
                perform_movies_qa(llm, question, edge_dict_list)

            
            # Matcher Module
            # edge_description_list = remove_element(edge_description_list, "1990")
            matched_cot_list, context_to_cot_match = get_used_sentences_ids(edge_description_list, cot_response_list)
            subgraph_elements_list = []

            subgraph_elements_list = build_cyto_subgraph_elements_list(seed_nodes, edge_dict_list, context_to_cot_match, final_answer_list)
            
            # subgraph_elements_list = build_cyto_subgraph_elements_list(edge_dict_list, [], response_list)

            doc_section = build_edge_description_table(edge_description_list)
            subgraph_section = draw_subgraph(subgraph_elements_list, SUBGRAPH_FIGURE_ID)
        
            QA_section = build_qa_info_table(instruction_msg, prompt, llm_response)
            llm_answers_section = build_llm_answers_table(final_answer_list)
            llm_cot_section = build_llm_cot_table(matched_cot_list)

        elif kg_name == "umls":
            # edge_dict_list, entities_set, edge_description_list, instruction_msg, prompt, llm_response, cot_response_list, llm_asnwers, response_list = \
            perform_umls_qa(llm, question, depth = int(hops_count))

            # # used_entities_id = get_used_entities_ids(entities_set, response_list)
            # used_sentences_id = get_used_sentences_ids(edge_description_list, cot_response_list)
            # subgraph_elements_list = []

            # if cot:
            #     subgraph_elements_list = build_cyto_subgraph_elements_list(edge_dict_list, used_sentences_id, response_list)
            # else:
            #     subgraph_elements_list = build_cyto_subgraph_elements_list(edge_dict_list, [], response_list)

            # doc_section = build_edge_description_table(edge_description_list)
            # subgraph_section = draw_subgraph(subgraph_elements_list, SUBGRAPH_FIGURE_ID)
        
            # QA_section = build_qa_info_table(instruction_msg, prompt, llm_response)
            # llm_response_section = build_llm_response_table(response_list)

        else:
            print("Invalid KG name")

    return doc_section, subgraph_section, QA_section, llm_answers_section, llm_cot_section
