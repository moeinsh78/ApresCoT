import dash
from services.examplesAssets import (
    get_toy_form_for_use_case,
    OUTPUT_LISTS
)

from typing import Optional
from dash import Input, Output, State, callback, ctx
from services.components import (
    build_page_layout,
    build_edge_description_table,
    build_llm_answers_table,
    build_llm_cot_table,
    draw_subgraph
)


from services.config import (
    APP_NAME,
    HIDDEN_STYLE,
    VISIBLE_STYLE,
)

dash.register_page(__name__, title=APP_NAME)




HOP_SELECT_ID = "EXAMPLES_PAGE_HOP_SELECT_ID"
QUESTION_INPUT_ID = "EXAMPLES_PAGE_QUESTION_INPUT_ID"
LLM_SELECT_ID = "EXAMPLES_PAGE_LLM_SELECT_ID"
KG_SELECT_ID = "EXAMPLES_PAGE_KG_SELECT_ID"
GENERATE_BUTTON_ID = "EXAMPLES_PAGE_GENERATE_BUTTON_ID"
DESCRIPTION_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_DESCRIPTION_TABLE_CONTAINER_ID"
SUBGRAPH_CONTAINER_ID = "EXAMPLES_PAGE_SUBGRAPH_CONTAINER_ID"
SUBGRAPH_FIGURE_ID = "EXAMPLES_PAGE_SUBGRAPH_FIGURE_ID"
QA_INFO_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_INFO_TABLE_CONTAINER_ID"
LLM_ANSWERS_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_LLM_ANSWERS_TABLE_CONTAINER_ID"
LLM_COT_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_LLM_COT_TABLE_CONTAINER_ID"
RESULT_SECTION_ID = "EXAMPLES_RESULT_SECTION_ID"

USE_CASE_1_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_1_BUTTON_ID"
USE_CASE_2_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_2_BUTTON_ID"
USE_CASE_3_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_3_BUTTON_ID"



layout = build_page_layout(
    QUESTION_INPUT_ID, HOP_SELECT_ID, LLM_SELECT_ID, KG_SELECT_ID,
    DESCRIPTION_TABLE_CONTAINER_ID, SUBGRAPH_CONTAINER_ID,
    QA_INFO_TABLE_CONTAINER_ID, LLM_ANSWERS_TABLE_CONTAINER_ID, 
    LLM_COT_TABLE_CONTAINER_ID, GENERATE_BUTTON_ID, RESULT_SECTION_ID,

    use_case_1_btn_id=USE_CASE_1_BUTTON_ID, 
    use_case_2_btn_id=USE_CASE_2_BUTTON_ID, 
    use_case_3_btn_id=USE_CASE_3_BUTTON_ID
)


@callback(
    Output(QUESTION_INPUT_ID, "value"),
    Output(LLM_SELECT_ID, "value"),
    Output(KG_SELECT_ID, "value"),
    Output(USE_CASE_1_BUTTON_ID, "active"),
    Output(USE_CASE_2_BUTTON_ID, "active"),
    Output(USE_CASE_3_BUTTON_ID, "active"),
    Input(USE_CASE_1_BUTTON_ID, "n_clicks"),
    Input(USE_CASE_2_BUTTON_ID, "n_clicks"),
    Input(USE_CASE_3_BUTTON_ID, "n_clicks")
)
def on_click_use_case_button(
    uc1_n_clicks: Optional[int], uc2_n_clicks: Optional[int], uc3_n_clicks: Optional[int]
):
    use_case_num = 1
    if ctx.triggered_id == USE_CASE_2_BUTTON_ID:
        use_case_num = 2
    if ctx.triggered_id == USE_CASE_3_BUTTON_ID:
        use_case_num = 3

    question, llm, kg = get_toy_form_for_use_case(use_case_num)
    print("Question:", question)
    print("LLM:", llm)
    print("KG:", kg)
    return question, llm, kg, use_case_num == 1, use_case_num == 2, use_case_num == 3



@callback(
    # Output(DESCRIPTION_TABLE_CONTAINER_ID, "children"),
    Output(SUBGRAPH_CONTAINER_ID, "children"),
    # Output(QA_INFO_TABLE_CONTAINER_ID, "children"),
    Output(LLM_ANSWERS_TABLE_CONTAINER_ID, "children"),
    Output(LLM_COT_TABLE_CONTAINER_ID, "children"),
    Output(RESULT_SECTION_ID, "style"),
    Input(GENERATE_BUTTON_ID, "n_clicks"),
    State(USE_CASE_1_BUTTON_ID, "active"),
    State(USE_CASE_2_BUTTON_ID, "active"),
    State(USE_CASE_3_BUTTON_ID, "active"),
)
def on_generate(
    get_n_clicks: Optional[int], uc1_active: bool, uc2_active: bool, uc3_active: bool
):
    input_is_invalid = get_n_clicks is None

    doc_section, subgraph_section = None, None
    QA_section, llm_answers_section, llm_cot_section = None, None, None
    results_style = HIDDEN_STYLE

    if not input_is_invalid:
        # Set current use case number
        use_case_num = 1
        if uc2_active:
            use_case_num = 2
        if uc3_active:
            use_case_num = 3

        desired_use_case_outputs = OUTPUT_LISTS[use_case_num-1]
        # doc_section = build_edge_description_table(desired_use_case_outputs["EDGE_DESCS"])
        subgraph_section = draw_subgraph(desired_use_case_outputs["GRAPH_ELEMENTS"], SUBGRAPH_FIGURE_ID)
            
        # QA_section = None
        llm_answers_section = build_llm_answers_table(desired_use_case_outputs["ANSWERS"])
        llm_cot_section = build_llm_cot_table(desired_use_case_outputs["COT_STEPS"])
        
        results_style = VISIBLE_STYLE

    return subgraph_section, llm_answers_section, llm_cot_section, results_style
