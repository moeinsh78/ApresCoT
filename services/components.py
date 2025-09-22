import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Optional, Any, Dict, List
from collections import Counter
from dash import dcc, html
import dash_cytoscape as cyto

from services.config import (
    HIDDEN_STYLE, 
    LLM_OPTIONS,
    KG_OPTIONS,
)


TABLE_BORDERED = True
TABLE_STRIPED = False
TABLE_HOVER = True
TABLE_WHITESPACE_STYLE = "pre-line"
TABLE_SIZE = "sm"
TABLE_BORDER_SIZE = "5px"
TABLE_FONT_SIZE = 16
# TABLE_STYLE = { "font-size": TABLE_FONT_SIZE }

GRAPH_EDGE_FONT_SIZE = "4px"
GRAPH_NODE_FONT_SIZE = "4px"

SUBGRAPH_HEIGHT = "650px"
SUBGRAPH_NODE_COLOR = "#D3D3D3"
SUBGRAPH_EDGE_COLOR = "#D3D3D3"
GREEN_COT_COLOR = "#ADEBB3"
LIGHT_GREEN_COT_COLOR = "#68BA7F" 
DARK_GREEN_COT_COLOR = "#9EB8A0" # Good
BLUE_COT_COLOR = "#66DE78"
LIGHT_BLUE_COT_COLOR = "#54B84D"
SOURCE_NODE_COLOR = "#182E6F"
RESPONSE_NODE_COLOR = "#8FD9FB"
SUBGRAPH_NODE_POSITIONING = "cose"
NO_MATCH_COLOR = "#FFFFFF"


TABLE_STYLE = {
    "font-size": TABLE_FONT_SIZE, 
    "border": TABLE_BORDER_SIZE,
    "whiteSpace": TABLE_WHITESPACE_STYLE
}

PERM_TYPE_ALL = 0
PERM_TYPE_RANDOM = 1
PERM_TYPE_REL = 2

COMB_SAMPLE_TYPE_ALL = 0
COMB_SAMPLE_TYPE_RANDOM = 1

POS_DIST_TYPE_V = 0
POS_DIST_TYPE_UNIFORM = 1

COMB_SEARCH_TYPE_BU = 0
COMB_SEARCH_TYPE_TD = 1

REL_SCORING_TYPE_NONE = 0
REL_SCORING_TYPE_RETR_SCORES = 1
REL_SCORING_TYPE_ATTN_SCORES = 2

# Map each relevance scoring type to a SourceScoringStrategy
REL_SCORING_TYPE_TO_STRATEGY = [
    REL_SCORING_TYPE_NONE, REL_SCORING_TYPE_RETR_SCORES, REL_SCORING_TYPE_ATTN_SCORES
]




def build_table(table_header: Any, table_body: Any, bordered: bool = TABLE_BORDERED,
                striped: bool = TABLE_STRIPED, hover: bool = TABLE_HOVER,
                size: str = TABLE_SIZE, style: Dict[str, Any] = TABLE_STYLE) -> dbc.Table:
    """
    Build a dbc.Table using a table header and body.
    """
    return dbc.Table(
        table_header + table_body,
        bordered=bordered,
        striped=striped,
        hover=hover,
        size=size,
        style=style,
    )



def build_edge_description_table(edge_descriptions: List[str]) -> dbc.Table:
    """
    Build a table that displays a document ranking.
    """
    return build_table(
        [html.Thead(html.Tr([html.Th(h) for h in ["ID", "Edge Description"]]))],
        [html.Tbody([
            html.Tr([
                html.Td(f"({i + 1})"),
                html.Td(desc),
            ])
            for i, desc in enumerate(edge_descriptions)
        ])],
        style=TABLE_STYLE,
    )



def build_qa_info_table(prompt: str, llm_response: str, llm_final_answers: List[str], llm_cot: List[str]) -> dbc.Table:
    """
    Build a table that displays the llm's prompting details.
    """
    parsed_llm_response = "Steps:\n" + "\n".join(llm_cot) + "\n\nAnswers:\n" + "\n".join(llm_final_answers)

    return build_table(
        [],
        [html.Tbody([
            html.Tr([
                html.Td("LLM Prompt"),
                html.Td(prompt),
            ]),
            html.Tr([
                html.Td("LLM Response"),
                html.Td(llm_response),
            ]), 
            html.Tr([
                html.Td("LLM Parsed Response"),
                html.Td(parsed_llm_response),
            ]),
        ])],
        style=TABLE_STYLE,
    )


def build_llm_answers_table(node_to_answer_match: List[Dict]) -> dbc.Table:
    """
    Build a table that displays the response items that the llm has given.
    """
    colors = []
    for ans_info in node_to_answer_match:
        if ans_info["Index"] == "No Match":
            colors.append(NO_MATCH_COLOR)
        else: 
            colors.append(RESPONSE_NODE_COLOR)
    
    return build_table(
        [html.Thead(html.Tr([html.Th(h) for h in ["ID", "LLM Answer Item"]]))],
        [html.Tbody([
            html.Tr([
                html.Td(
                    f"[A{i + 1}]",
                    style={"background-color": colors[i]},
                ),
                html.Td(ans["Answer"]),
            ])
            for i, ans in enumerate(node_to_answer_match)
        ])],
        style=TABLE_STYLE,
    )



def build_llm_cot_table(cot_match_dicts: List[Dict]) -> dbc.Table:
    """
    Build a table that displays the pieces of information in the reasoning text generate by the LLM.
    """
    colors = []
    # index = 0
    for cot_step_info in cot_match_dicts:
        # index += 1

        # print("Index:", index, cot_step_info)
        if cot_step_info["Most Similar Context ID"] == "No Match":
            colors.append(NO_MATCH_COLOR)
        else: 
            colors.append(GREEN_COT_COLOR)

    return build_table(
        # , "Matched Edge Number"
        [html.Thead(html.Tr([html.Th(h) for h in ["ID", "LLM Reasoning Step"]]))],
        [html.Tbody([
            html.Tr([
                html.Td(
                    f"[S{i + 1}]",
                    style={"background-color": colors[i]},
                ),
                html.Td(cot_step_info["COT Step"]),
            ])
            for i, cot_step_info in enumerate(cot_match_dicts)
        ])],
        style=TABLE_STYLE,
    )



def build_edge_description_section(descriptions_table_id: str) -> html.Div:
    """
    Build a section to display the edge descriptions of the subgraph 
    extracted from the knowledge graph.
    """
    return html.Div(
        dbc.Row(
            dbc.Col(
                [
                    html.H5("Subgraph Edge Descriptions"),
                    html.Div(id=descriptions_table_id)
                ],
                xs=12,
                sm=7
            )
        )
    )


def build_qa_info_section(qa_info_table_id: str) -> html.Div:
    return html.Div(
        dbc.Row(
            dbc.Col(
                [
                    html.H5("QA Information Table"),
                    html.Div(id=qa_info_table_id)
                ],
                xs=12
            )
        )
    )


def build_llm_response_section(llm_answers_table_id: str, llm_cot_table_id: str) -> html.Div:
    return html.Div(
        dbc.Row([
            dbc.Col(
                [
                    html.H5("LLM Final Answers"),
                    html.Div(id=llm_answers_table_id)
                ],
                xs=12,
                sm=3
            ), 
            dbc.Col(
                [
                    html.H5("LLM Reasoning Steps"),
                    html.Div(id=llm_cot_table_id)
                ],
                xs=12,
                sm=9
            )] 
        )
    )


def build_subgraph_section(subgraph_container_id: str) -> html.Div:
    return html.Div(
        dbc.Row(
            dbc.Col(
                [
                    html.H5("Neighboring Subgraph"),
                    html.Div(id=subgraph_container_id)
                ],
                xs=12,
            )
        ),
    )



def draw_subgraph(subgraph_elements_list: List[Dict], subgraph_figure_id: str) -> html.Div:
    """
    Draws an interactive knowledge graph representing the neighboring subgraph
    of the knowledge graph node that holds the answer to the asked question. 
    """
    return html.Div([
        cyto.Cytoscape(
            id=subgraph_figure_id, 
            elements=subgraph_elements_list,
            layout={"name": SUBGRAPH_NODE_POSITIONING},
            style={
                "height": SUBGRAPH_HEIGHT
            },
            stylesheet= [
                {
                    "selector": "node",
                    "style": {
                        'label': 'data(label)',
                        "font-size": GRAPH_NODE_FONT_SIZE,
                        'width': '10px', 
                        'height': '10px', 
                    }
                },
                {
                    "selector": ".response",
                    "style": {
                        "background-color": RESPONSE_NODE_COLOR,
                    }
                },
                {
                    "selector": ".source",
                    "style": {
                        "background-color": SOURCE_NODE_COLOR,
                    }
                },
                {
                    "selector": ".normal",
                    "style": {
                        "background-color": SUBGRAPH_NODE_COLOR,
                    }
                },
                {
                    "selector": "edge",
                    "style": {
                        "label": "data(weight)",
                        "line-color": SUBGRAPH_EDGE_COLOR,
                        "font-size": GRAPH_EDGE_FONT_SIZE,
                        "width": 1.5
                    }
                }, 
                {
                    "selector": ".curved",
                    "style": {
                        "curve-style": "bezier",
                        "control-point-step-size": 20,
                    }
                }, 
                {
                    'selector': ".cot-edge",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
            ]
        ),
        ], 
        style={
            "border": "2px solid black",
            "width": "90%",
        }
    )


def build_demo_welcome_alert() -> dbc.Alert:
    """
    Build a "Welcome" alert welcoming users to the AprèsCoT's demo page.
    """
    return dbc.Alert(
        [
            html.H4("Welcome to the AprèsCoT's demo page!"),
            html.Hr(),
            html.P(
                (
                    "On this page, you may interact with AprèsCoT. "
                    "You may enter a question, choose a Large Language Model to answer your question, "
                    "and pick a knowledge graph (KG) to ground the LLM's response. Upon clicking \"Generate\", "
                    "the selected LLM will be prompted to answer your question, and the response and " 
                    "reasoning steps will be mapped onto the chosen knowledge graph. "
                    "\n\nBelow, you may find and use some sample questions with their corresponding KG to get started: \n"
                    "What other movies have the same director with the movie Inception? --> KG: MetaQA Movies \n"
                    "What were the release years of the films starred by Jean Rochefort? --> KG: MetaQA Movies \n"
                    "What types are the films starred by actors in The Exploding Girl? --> KG: MetaQA Movies \n"
                    "What types of animals are affected by dysfunctions caused by Fungus? --> KG: UMLS Relations \n"
                    "Which countries have land borders with Germany's neighbours? --> KG: WikiData \n"
                ),
                className="mb-0",
            ),
        ],
        color="success",
        style={"whiteSpace": "pre-line"},
        dismissable=True
    )


def build_examples_welcome_alert() -> dbc.Alert:
    """
    Build a "Welcome" alert welcoming users to the AprèsCoT's examples page.
    """
    return dbc.Alert(
        [
            html.H4("Welcome to AprèsCoT!"),
            html.Hr(),
            html.P(
                (
                    "In this page, you may explore the use cases described in the paper, "
                    "and interact with sample explanations. To generate these explanations, "
                    "select one of the use case buttons below, and then click "
                    "\"Generate\". You may also interact with the subgraph by dragging and "
                    "repositioning nodes and edges to view the knowledge graph labels and connections. "
                    "When you are ready, we invite you to proceed to the full demo by "
                    "selecting the \"Full Demo\" button in the navigation bar. Thanks!"
                ),
                className="mb-0",
            ),
        ],
        color="success",
        style={"whiteSpace": "pre-line"},
        dismissable=True
    )



def build_form_section(
    question_input_id: str, hop_select_id: str, llm_select_id: str, kg_select_id: str,
    generate_btn_id: str, is_demo_mode: bool = False, 
    llms: List[Dict[str, str]] = LLM_OPTIONS,
    knowledge_graphs: List[Dict[str, str]] = KG_OPTIONS,

    use_case_1_btn_id: Optional[str] = None,
    use_case_2_btn_id: Optional[str] = None,
    use_case_3_btn_id: Optional[str] = None

) -> html.Div:
    """
    Build the form section of the page, containing welcome alert / use case buttons
    (as needed), common form elements, and permutations / combinations forms
    (under their respective tabs).
    """
    
    combined_form = html.Div(dbc.Card(dbc.Stack(
        [
            dbc.Row(
                [dbc.Col(
                    [
                        dbc.Label("Question"),
                        dbc.Input(
                            id=question_input_id,
                            type="text",
                            placeholder="Enter your question",
                            maxlength=100,
                            disabled=is_demo_mode
                        ),
                    ],
                    sm=6,
                    style={"padding": 10}
                ),
                dbc.Col(
                    [
                        dbc.Label("LLM"),
                        dbc.Select(
                            id=llm_select_id,
                            options=llms,
                            value=llms[0]["value"],
                            disabled=is_demo_mode
                        )
                    ],
                    sm=3,
                    style={"padding": 10}
                ),
                dbc.Col(
                    [
                        dbc.Label("Knowledge Graph"),
                        dbc.Select(
                            id=kg_select_id,
                            options=knowledge_graphs,
                            value=knowledge_graphs[0]["value"],
                            disabled=is_demo_mode
                        )
                    ],
                    sm=3,
                    style={"padding": 10}
                )]
            ),
            dbc.Col(
                dbc.Button(
                    "Generate",
                    id=generate_btn_id,
                    color="primary"
                ),
                align="center",
                sm=2
            )
        ],
        direction="vertical",
        gap=3,
        style={"padding": 24, "background-color": "whitesmoke"},
    )))
    
    
    if is_demo_mode:
        return html.Div(dbc.Stack(
            [
                build_examples_welcome_alert(),
                html.Div([
                    dbc.Button(
                        "Use Case 1: Consistent Answers and CoT",
                        id=use_case_1_btn_id,
                        outline=True, color="primary", className="me-1"
                    ),
                    dbc.Button(
                        "Use Case 2: Knowledge Graph Data Quality",
                        id=use_case_2_btn_id,
                        outline=True, color="primary", className="me-1"
                    ),
                    dbc.Button(
                        "Use Case 3: Inconsistent Answers and CoT",
                        id=use_case_3_btn_id,
                        outline=True, color="primary", className="me-1"
                    ),
                ]),
                combined_form
            ],
            gap=2,
        ))
    else:
        return html.Div(dbc.Stack(
            [
                build_demo_welcome_alert(),
                combined_form
            ],
            gap=2,
        ))


def build_page_layout(
    question_input_id: str, hop_select_id: str, llm_select_id: str, 
    kg_select_id: str, description_table_id: str, subgraph_container_id: str, 
    qa_info_table_container_id: str, llm_answers_table_container_id: str, 
    llm_cot_table_container_id: str, generate_btn_id: str, results_id: str,
    llms: List[Dict[str, str]] = LLM_OPTIONS,
    kgs: List[Dict[str, str]] = KG_OPTIONS,

    use_case_1_btn_id: Optional[str] = None,
    use_case_2_btn_id: Optional[str] = None,
    use_case_3_btn_id: Optional[str] = None
):
    """
    Build a basic page layout with a form and edge descriptions table.
    """
    is_demo_mode = use_case_1_btn_id is not None


    form_section = build_form_section(
        question_input_id, hop_select_id, llm_select_id, 
        kg_select_id, generate_btn_id, is_demo_mode, 
        llms, kgs,
        use_case_1_btn_id, use_case_2_btn_id, use_case_3_btn_id
    )

    qa_info_section = build_qa_info_section(qa_info_table_container_id)
    subgraph_section = build_subgraph_section(subgraph_container_id)
    llm_response_section = build_llm_response_section(llm_answers_table_container_id, llm_cot_table_container_id)


    return dbc.Stack(
        [
            html.Div([
                form_section,
            ]),
            html.Div(
                dbc.Stack(
                    [llm_response_section, subgraph_section, qa_info_section],
                    gap=5
                ),
                id=results_id,
                style=HIDDEN_STYLE
            )
            # html.Div(
            #     form_section,
            # ),
            # html.Div(
            #     llm_response_section
            # ),
            # html.Div(
            #     subgraph_section
            # ),
            # html.Div(
            #     edge_desc_section
            # ),
            # html.Div(
            #     qa_info_section
            # ),
        ],
        gap=5,
        style={"padding": 20}
    )
