import dash
from services.examplesAssets import (
    get_toy_form_for_use_case,
    OUTPUT_LISTS
)
from typing import Optional
from dash import Input, Output, State, callback, ctx
from services.components import (
    # EMPTY_PIE_GRAPH_FIG, 
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




# GENERATE_PERMS_BUTTON_ID = "EXAMPLES_PAGE_GENERATE_PERMS_BUTTON_ID"
# GENERATE_COMBS_BUTTON_ID = "EXAMPLES_PAGE_GENERATE_COMBS_BUTTON_ID"
# FORM_TABS_ID = "EXAMPLES_PAGE_FORM_TABS_ID"
# QUESTION_INPUT_ID = "EXAMPLES_PAGE_QUESTION_INPUT_ID"
# K_INPUT_ID = "EXAMPLES_PAGE_K_INPUT_ID"
# LLM_SELECT_ID = "EXAMPLES_PAGE_LLM_SELECT_ID"
# INDEX_SELECT_ID = "EXAMPLES_PAGE_INDEX_SELECT_ID"
# MAX_ANSWER_TOKENS_INPUT_ID = "EXAMPLES_PAGE_MAX_ANSWER_TOKENS_INPUT_ID"
# MAX_COUNTERFACTUAL_INPUT_ID = "EXAMPLES_PAGE_MAX_COUNTERFACTUAL_INPUT_ID"
# NUM_PERMUTATIONS_INPUT_ID = "EXAMPLES_PAGE_NUM_PERMUTATIONS_INPUT_ID"
# NUM_COMBINATIONS_INPUT_ID = "EXAMPLES_PAGE_NUM_COMBINATIONS_INPUT_ID"
# BATCH_SIZE_INPUT_ID = "EXAMPLES_PAGE_BATCH_SIZE_INPUT_ID"
# RESULTS_CONTAINER_ID = "EXAMPLES_PAGE_RESULTS_CONTAINER_ID"
# RANKING_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_RANKING_TABLE_CONTAINER_ID"
# ANSWER_GROUPS_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_ANSWER_GROUPS_TABLE_CONTAINER_ID"
# ANSWER_RULES_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_ANSWER_RULES_TABLE_CONTAINER_ID"
# PERM_TYPE_RADIO_ID = "EXAMPLES_PAGE_PERM_TYPE_RADIO_ID"
# COMB_SAMPLING_TYPE_RADIO_ID = "EXAMPLES_PAGE_COMB_SAMPLING_TYPE_RADIO_ID"
# POS_DIST_TYPE_RADIO_ID = "EXAMPLES_PAGE_POS_DIST_TYPE_RADIO_ID"
# REL_SCORING_TYPE_PERMS_RADIO_ID = "EXAMPLES_PAGE_REL_SCORING_TYPE_PERMS_RADIO_ID"
# REL_SCORING_TYPE_COMBS_RADIO_ID = "EXAMPLES_PAGE_REL_SCORING_TYPE_COMBS_RADIO_ID"
# COREF_TOGGLE_ID = "EXAMPLES_PAGE_COREF_TOGGLE_ID"
# COMB_SEARCH_TYPE_RADIO_ID = "EXAMPLES_PAGE_COMB_SEARCH_TYPE_RADIO_ID"
# ANSWER_PIE_CHART_ID = "EXAMPLES_PAGE_ANSWER_PIE_CHART_ID"
# COUNTERFACTUALS_TABLE_CONTAINER_ID = "EXAMPLES_PAGE_COUNTERFACTUALS_TABLE_CONTAINER_ID"
# ERROR_TOAST_ID = "EXAMPLES_PAGE_ERROR_TOAST_ID"
# REFERENCE_PRED_LABEL_ID = "EXAMPLES_PAGE_REFERENCE_PRED_LABEL_ID"

# USE_CASE_1_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_1_BUTTON_ID"
# USE_CASE_2_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_2_BUTTON_ID"
# USE_CASE_3_BUTTON_ID = "EXAMPLES_PAGE_USE_CASE_3_BUTTON_ID"


layout = build_page_layout(
    QUESTION_INPUT_ID, HOP_SELECT_ID, LLM_SELECT_ID, KG_SELECT_ID,
    DESCRIPTION_TABLE_CONTAINER_ID, SUBGRAPH_CONTAINER_ID,
    QA_INFO_TABLE_CONTAINER_ID, LLM_ANSWERS_TABLE_CONTAINER_ID, 
    LLM_COT_TABLE_CONTAINER_ID, GENERATE_BUTTON_ID, RESULT_SECTION_ID,

    use_case_1_btn_id=USE_CASE_1_BUTTON_ID, 
    use_case_2_btn_id=USE_CASE_2_BUTTON_ID, 
    use_case_3_btn_id=USE_CASE_3_BUTTON_ID
)


# layout = build_page_layout(
#     ERROR_TOAST_ID, QUESTION_INPUT_ID, LLM_SELECT_ID, INDEX_SELECT_ID, K_INPUT_ID,
#     BATCH_SIZE_INPUT_ID, MAX_ANSWER_TOKENS_INPUT_ID, MAX_COUNTERFACTUAL_INPUT_ID,
#     REL_SCORING_TYPE_PERMS_RADIO_ID, POS_DIST_TYPE_RADIO_ID, PERM_TYPE_RADIO_ID,
#     NUM_PERMUTATIONS_INPUT_ID, COREF_TOGGLE_ID, GENERATE_PERMS_BUTTON_ID,
#     REL_SCORING_TYPE_COMBS_RADIO_ID, COMB_SEARCH_TYPE_RADIO_ID,
#     COMB_SAMPLING_TYPE_RADIO_ID, NUM_COMBINATIONS_INPUT_ID, GENERATE_COMBS_BUTTON_ID,
#     FORM_TABS_ID, RANKING_TABLE_CONTAINER_ID, RESULTS_CONTAINER_ID, ANSWER_PIE_CHART_ID,
#     ANSWER_GROUPS_TABLE_CONTAINER_ID, ANSWER_RULES_TABLE_CONTAINER_ID,
#     COUNTERFACTUALS_TABLE_CONTAINER_ID, REFERENCE_PRED_LABEL_ID,
#     indexes=TOY_INDEXES,
#     default_num_perms=10,
#     default_num_combs=10,
#     default_top_k=5,
#     default_batch_size=1,
#     default_max_tokens=10,
#     default_max_cf_tests=10,
#     use_case_1_btn_id=USE_CASE_1_BUTTON_ID,
#     use_case_2_btn_id=USE_CASE_2_BUTTON_ID,
#     use_case_3_btn_id=USE_CASE_3_BUTTON_ID
# )


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
    return question, llm, kg, use_case_num == 1, use_case_num == 2, use_case_num == 3



@callback(
    Output(DESCRIPTION_TABLE_CONTAINER_ID, "children"),
    Output(SUBGRAPH_CONTAINER_ID, "children"),
    Output(QA_INFO_TABLE_CONTAINER_ID, "children"),
    Output(LLM_ANSWERS_TABLE_CONTAINER_ID, "children"),
    Output(LLM_COT_TABLE_CONTAINER_ID, "children"),
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

    if not input_is_invalid:
        # Set current use case number
        use_case_num = 1
        if uc2_active:
            use_case_num = 2
        if uc3_active:
            use_case_num = 3

        desired_use_case_outputs = OUTPUT_LISTS[use_case_num-1]
        doc_section = build_edge_description_table(desired_use_case_outputs["EDGE_DESCS"])
        subgraph_section = draw_subgraph(desired_use_case_outputs["GRAPH_ELEMENTS"], SUBGRAPH_FIGURE_ID)
            
        QA_section = None
        llm_answers_section = build_llm_answers_table(desired_use_case_outputs["ANSWERS"])
        llm_cot_section = build_llm_cot_table(desired_use_case_outputs["COT_STEPS"])

    return doc_section, subgraph_section, QA_section, llm_answers_section, llm_cot_section



# @callback(
#     Output(RANKING_TABLE_CONTAINER_ID, "children"),
#     Output(ANSWER_GROUPS_TABLE_CONTAINER_ID, "children"),
#     Output(ANSWER_RULES_TABLE_CONTAINER_ID, "children"),
#     Output(ANSWER_PIE_CHART_ID, "figure"),
#     Output(COUNTERFACTUALS_TABLE_CONTAINER_ID, "children"),
#     Output(REFERENCE_PRED_LABEL_ID, "children"),
#     Output(ERROR_TOAST_ID, "is_open"),
#     Output(ERROR_TOAST_ID, "children"),
#     Output(RESULTS_CONTAINER_ID, "style"),
#     Input(GENERATE_PERMS_BUTTON_ID, "n_clicks"),
#     Input(GENERATE_COMBS_BUTTON_ID, "n_clicks"),
#     State(USE_CASE_1_BUTTON_ID, "active"),
#     State(USE_CASE_2_BUTTON_ID, "active"),
#     State(USE_CASE_3_BUTTON_ID, "active"),
# )
# def on_generate(
#     gen_perms_n_clicks: Optional[int], gen_combs_n_clicks: Optional[int],
#     uc1_active: bool, uc2_active: bool, uc3_active: bool
# ):
#     input_is_invalid = gen_perms_n_clicks is None and gen_combs_n_clicks is None

#     ranking_table, groups_table, rules_table = None, None, None
#     pie_chart, counterfactuals_table = EMPTY_PIE_GRAPH_FIG, None
#     reference_pred = "Reference: N/A"
#     err_msg, results_style = None, HIDDEN_STYLE

#     if not input_is_invalid:
#         # Set current use case number
#         use_case_num = 1
#         if uc2_active:
#             use_case_num = 2
#         if uc3_active:
#             use_case_num = 3

#         is_permutations_mode = ctx.triggered_id == GENERATE_PERMS_BUTTON_ID
#         (
#             ranking_table,
#             groups_table,
#             rules_table,
#             pie_chart,
#             counterfactuals_table,
#             reference_pred
#         ) = get_toy_insights_for_use_case(use_case_num, is_permutations_mode)
#         err_msg, results_style = None, VISIBLE_STYLE

#     return (
#         ranking_table, groups_table, rules_table, pie_chart, counterfactuals_table,
#         reference_pred, err_msg is not None, err_msg, results_style
#     )
