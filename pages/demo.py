import dash
from typing import List, Optional
from dash import Input, Output, State, callback, ctx
from services.components import (
    build_page_layout,
    build_edge_description_table, 
    build_qa_info_table, 
    build_llm_answers_table,
    build_llm_cot_table,
    draw_subgraph
)
from services.config import (
    APP_NAME, 
    LLM_NAMES, 
    SYSTEM_NAMES,
    VISIBLE_STYLE,
    HIDDEN_STYLE,
)

from src.aprescot.qa import perform_qa

dash.register_page(__name__, title=APP_NAME)



HOP_SELECT_ID = "DEMO_PAGE_HOP_SELECT_ID"
QUESTION_INPUT_ID = "DEMO_PAGE_QUESTION_INPUT_ID"
LLM_SELECT_ID = "DEMO_PAGE_LLM_SELECT_ID"
KG_SELECT_ID = "DEMO_PAGE_KG_SELECT_ID"
GENERATE_BUTTON_ID = "DEMO_PAGE_GENERATE_BUTTON_ID"
DESCRIPTION_TABLE_CONTAINER_ID = "DEMO_PAGE_DESCRIPTION_TABLE_CONTAINER_ID"
SUBGRAPH_CONTAINER_ID = "DEMO_PAGE_SUBGRAPH_CONTAINER_ID"
SUBGRAPH_FIGURE_ID = "DEMO_PAGE_SUBGRAPH_FIGURE_ID"
QA_INFO_TABLE_CONTAINER_ID = "DEMO_PAGE_INFO_TABLE_CONTAINER_ID"
LLM_ANSWERS_TABLE_CONTAINER_ID = "DEMO_PAGE_LLM_ANSWERS_TABLE_CONTAINER_ID"
LLM_COT_TABLE_CONTAINER_ID = "DEMO_PAGE_LLM_COT_TABLE_CONTAINER_ID"
RESULT_SECTION_ID = "DEMO_RESULT_SECTION_ID"


layout = build_page_layout(
    QUESTION_INPUT_ID, HOP_SELECT_ID, LLM_SELECT_ID, KG_SELECT_ID,
    DESCRIPTION_TABLE_CONTAINER_ID, SUBGRAPH_CONTAINER_ID,
    QA_INFO_TABLE_CONTAINER_ID, LLM_ANSWERS_TABLE_CONTAINER_ID, 
    LLM_COT_TABLE_CONTAINER_ID, GENERATE_BUTTON_ID, RESULT_SECTION_ID
)



@callback(
    # Output(DESCRIPTION_TABLE_CONTAINER_ID, "children"),
    Output(SUBGRAPH_CONTAINER_ID, "children"),
    # Output(QA_INFO_TABLE_CONTAINER_ID, "children"),
    Output(LLM_ANSWERS_TABLE_CONTAINER_ID, "children"),
    Output(LLM_COT_TABLE_CONTAINER_ID, "children"),
    Output(RESULT_SECTION_ID, "style"),
    Input(GENERATE_BUTTON_ID, 'n_clicks'),    
    State(QUESTION_INPUT_ID, "value"),
    State(LLM_SELECT_ID, "value"),
    State(KG_SELECT_ID, "value"),
)
def on_generate(gen_btn_n_clicks: int, question: str, supported_qa_model: str, kg_name: str):
    input_is_invalid = (
        question is None
        or kg_name is None
        or len(question) == 0
        or len(kg_name) == 0
    )

    doc_section, subgraph_section = None, None
    QA_section, llm_answers_section, llm_cot_section = None, None, None
    results_style = HIDDEN_STYLE
    hops_count = 2

    if not input_is_invalid:
        system = SYSTEM_NAMES[supported_qa_model]
        llm = LLM_NAMES[supported_qa_model]
        print("REQUESTED LLM:", llm)
        print("REQUESTED system:", system)
        
        rag_enabled = not (system == "vanilla-gpt-3.5" or system == "vanilla-gpt-4o-mini")

        instruction_msg, prompt, llm_response, subgraph_edge_desc_list, node_to_answer_match, cot_match_dicts, subgraph_elements_list = \
            perform_qa(llm, kg_name, question, rag_enabled)
        

        print("Node to Answer Match:\n", node_to_answer_match)

        print("CoT to Edge Match:\n", cot_match_dicts)
        print("Subgraph Edge Description List:\n")
        for edge_desc in subgraph_edge_desc_list:
            print(edge_desc)

        print("Subgraph Elements:\n")
        for element in subgraph_elements_list:
            print(element)
        
        # doc_section = build_edge_description_table(subgraph_edge_desc_list)
        subgraph_section = draw_subgraph(subgraph_elements_list, SUBGRAPH_FIGURE_ID)
        
        # QA_section = build_qa_info_table(instruction_msg, prompt, llm_response)
        llm_answers_section = build_llm_answers_table(node_to_answer_match)
        llm_cot_section = build_llm_cot_table(cot_match_dicts)

        results_style = VISIBLE_STYLE

    return subgraph_section, llm_answers_section, llm_cot_section, results_style




# @callback(
#     # Output(DESCRIPTION_TABLE_CONTAINER_ID, "children"),
#     Output(SUBGRAPH_CONTAINER_ID, "children"),
#     # Output(QA_INFO_TABLE_CONTAINER_ID, "children"),
#     Output(LLM_ANSWERS_TABLE_CONTAINER_ID, "children"),
#     Output(LLM_COT_TABLE_CONTAINER_ID, "children"),
#     Output(RESULT_SECTION_ID, "style"),
#     Input(GENERATE_BUTTON_ID, 'n_clicks'),    
#     State(QUESTION_INPUT_ID, "value"),
#     State(LLM_SELECT_ID, "value"),
#     State(KG_SELECT_ID, "value"),
# )
# def on_generate(gen_btn_n_clicks: int, question: str, supported_qa_model: str, kg_name: str):
#     input_is_invalid = (
#         question is None
#         or kg_name is None
#         or len(question) == 0
#         or len(kg_name) == 0
#     )
#     input_is_invalid = gen_btn_n_clicks is None

#     doc_section, subgraph_section = None, None
#     QA_section, llm_answers_section, llm_cot_section = None, None, None
#     results_style = HIDDEN_STYLE

#     if not input_is_invalid:
#         # Set current example case number
#         use_case_num = gen_btn_n_clicks % OUTPUT_LISTS.__len__()
#         print("Use Case Number:", use_case_num)
#         sleep(3)

#         desired_use_case_outputs = OUTPUT_LISTS[use_case_num-1]
#         # doc_section = build_edge_description_table(desired_use_case_outputs["EDGE_DESCS"])
#         subgraph_section = draw_subgraph(desired_use_case_outputs["GRAPH_ELEMENTS"], SUBGRAPH_FIGURE_ID)
            
#         # QA_section = None
#         llm_answers_section = build_llm_answers_table(desired_use_case_outputs["ANSWERS"])
#         llm_cot_section = build_llm_cot_table(desired_use_case_outputs["COT_STEPS"])
        
#         results_style = VISIBLE_STYLE

#     return subgraph_section, llm_answers_section, llm_cot_section, results_style
