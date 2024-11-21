import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter
from dash import dcc, html
import dash_cytoscape as cyto

from services.config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_COUNTERFACTUALS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_PERMUTATIONS,
    DEFAULT_TOP_K,
    HIDDEN_STYLE,
    LLM_OPTIONS,
    KAGE_HOP_COUNT_OPTIONS,
    INDEX_OPTIONS, 
    KAGE_INDEX_OPTIONS,
)


TABLE_BORDERED = True
TABLE_STRIPED = True
TABLE_HOVER = True
TABLE_WHITESPACE_STYLE = "pre-line"
TABLE_SIZE = "sm"
TABLE_BORDER_SIZE = "5px"
TABLE_FONT_SIZE = 16
TABLE_STYLE = { "font-size": TABLE_FONT_SIZE }
SUBGRAPH_HEIGHT = "1200px"
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
NO_MATCH_COLOR = "#ED2100"


# ANSWERS_COLOR = [RESPONSE_NODE_COLOR, RESPONSE_NODE_COLOR]
# COT_COLOR = [GREEN_COT_COLOR, GREEN_COT_COLOR, GREEN_COT_COLOR, GREEN_COT_COLOR]

# ANSWERS_COLOR = [RESPONSE_NODE_COLOR, NO_MATCH_COLOR]
# COT_COLOR = [GREEN_COT_COLOR, GREEN_COT_COLOR, GREEN_COT_COLOR, NO_MATCH_COLOR]



COT_COLORS_MAPPING = [
    {"class": "color1", "code": "#ADEBB3"},
    {"class": "color2", "code": "#68BA7F"},
    {"class": "color3", "code": "#9EB8A0"},
    {"class": "color4", "code": "#66DE78"},
    {"class": "color5", "code": "#54B84D"},
]


KAGE_TABLE_STYLE = {
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
        style=KAGE_TABLE_STYLE,
    )



def build_qa_info_table(instruction_msg: str, prompt: str, llm_response: str) -> dbc.Table:
    """
    Build a table that displays the llm's prompting details.
    """
    return build_table(
        [],
        [html.Tbody([
            html.Tr([
                html.Td("LLM Instruction Message"),
                html.Td(instruction_msg),
            ]),
            html.Tr([
                html.Td("LLM Prompt"),
                html.Td(prompt),
            ]),
            html.Tr([
                html.Td("LLM Response"),
                html.Td(llm_response),
            ])
        ])],
        style=KAGE_TABLE_STYLE,
    )


def build_llm_answers_table(answers_list: List[str]) -> dbc.Table:
    """
    Build a table that displays the response items that the llm has given.
    """
    return build_table(
        [html.Thead(html.Tr([html.Th(h) for h in ["ID", "LLM Answer Item"]]))],
        [html.Tbody([
            html.Tr([
                html.Td(
                    f"[{i + 1}]",
                    style={"background-color": RESPONSE_NODE_COLOR},
                ),
                html.Td(desc),
            ])
            for i, desc in enumerate(answers_list)
        ])],
        style=KAGE_TABLE_STYLE,
    )



def build_llm_cot_table(cot_list: List[Dict]) -> dbc.Table:
    """
    Build a table that displays the response items that the llm has given.
    """
    return build_table(
        # , "Matched Edge Number"
        [html.Thead(html.Tr([html.Th(h) for h in ["ID", "LLM Chain of Thought Step"]]))],
        [html.Tbody([
            html.Tr([
                html.Td(
                    f"[{i + 1}]",
                    style={"background-color": GREEN_COT_COLOR},
                ),
                html.Td(cot_step_info["COT Step"]),
                # html.Td(
                #     "(" + str(cot_step_info["Most Similar Context ID"]) + ")",
                #     style={"background-color": GREEN_COT_COLOR},
                #     # style={"background-color": COT_COLORS_MAPPING[i % COT_COLORS_MAPPING.__len__()]["code"]},
                #     # style={"background-color": if (cot_step_info["Most Similar Context ID"] == "No Match") NO_MATCH_COT_COLOR else COT_COLORS_MAPPING[i % COT_COLORS_MAPPING.__len__()]["code"]},
                # ),
            ])
            for i, cot_step_info in enumerate(cot_list)
        ])],
        style=KAGE_TABLE_STYLE,
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
                    html.H5("KAGE QA Information Table"),
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
                    html.H5("LLM Chain of Thought"),
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
                        # "background-color": SUBGRAPH_NODE_COLOR,
                        "font-size": "4px",
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
                        "font-size": "3px",
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
                    'selector': ".color1",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
                {
                    'selector': ".color2",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
                {
                    'selector': ".color3",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
                {
                    'selector': ".color4",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
                {
                    'selector': ".color5",
                    'style': {
                        "line-color": LIGHT_GREEN_COT_COLOR,
                    }
                },
            ]
        ),
        ], 
        style={
            "border": "2px solid black",
            "width": "95%",
        }
    )


def build_kage_welcome_alert() -> dbc.Alert:
    """
    Build a "Welcome" alert welcoming users to the KAGE page.
    """
    return dbc.Alert(
        [
            html.H4("Welcome to AprèsCoT!"),
            html.Hr(),
            html.P(
                (
                    "On this page, you may explore the functionalities of AprèsCoT. "
                    "You may choose GPT-4o or ChatGPT 3.5 as your desired Large Language Model to answer "
                    "you question. Currently, the only knowledge graph to be used as the underlying data "
                    "structure for the QA task is MetaQA's movies knowledge graph. You may type in a " 
                    "question in the \"Question\" field with proper formatting and click the "
                    "\"Generate\" button, and the chatbot will be prompted with proper information "
                    "and your question. The responses and utilized prompts will be shown in "
                    "their corresponding tables. Below, you may find and use some sample questions "
                    "with proper formatting to which the LLM can respond: \n\n"
                    "In which movies have Gary Oldman and Tom Hardy both starred? --> Required Search Depth = 1 \n"
                    "What were the release years of the films starred by Jean Rochefort?"
                    " --> Required Search Depth = 2 \n"
                    "Who acted in the films written by Peter Yeldham? --> Required Search Depth = 2 \n"
                    "Who starred in The Dark Knight Rises? --> Required Search Depth = 1 \n"
                    "What genres are the films starred by Luke Kirby? --> Required Search Depth = 2 \n"
                    "What types are the films starred by actors in The Exploding Girl? --> Required Search Depth = 3 \n"
                    "When did the movies whose directors also directed Down Terrace release? --> Required Search Depth = 3 \n"
                    "What genres are the films starred by Alessandro Nivola? --> Required Search Depth = 2 \n"
                    "Which actors starred movies for the director of Muppets from Space? --> Required Search Depth = 3 \n"
                ),
                className="mb-0",
            ),
        ],
        color="success",
        style={"whiteSpace": "pre-line"},
        dismissable=True
    )



def build_kage_form_section(
    question_input_id: str, hop_select_id: str, llm_select_id: str, kg_select_id: str,
    generate_btn_id: str, llms: List[Dict[str, str]] = LLM_OPTIONS,
    hop_count_options: List[Dict[int, int]] = KAGE_HOP_COUNT_OPTIONS,
    knowledge_graphs: List[Dict[str, str]] = KAGE_INDEX_OPTIONS,
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
                            disabled=False
                        ),
                    ],
                    sm=6,
                ),
                # dbc.Col(
                #     [
                #         dbc.Label("Search Depth"),
                #         dbc.Select(
                #             id=hop_select_id,
                #             options=hop_count_options,
                #             value=hop_count_options[0]["value"],
                #             disabled=False
                #         )
                #     ],
                #     sm=2,
                # ),
                dbc.Col(
                    [
                        dbc.Label("QA System"),
                        dbc.Select(
                            id=llm_select_id,
                            options=llms,
                            value=llms[0]["value"],
                            disabled=False
                        )
                    ],
                    sm=3,
                ),
                dbc.Col(
                    [
                        dbc.Label("Knowledge Graph"),
                        dbc.Select(
                            id=kg_select_id,
                            options=knowledge_graphs,
                            value=knowledge_graphs[0]["value"],
                            disabled=False
                        )
                    ],
                    sm=3,
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
    

    return html.Div(dbc.Stack(
            [
                build_kage_welcome_alert(),
                combined_form
            ],
            gap=2,
        ))




def build_kage_page_layout(
    question_input_id: str, hop_select_id: str, llm_select_id: str, 
    kg_select_id: str, description_table_id: str, subgraph_container_id: str, 
    qa_info_table_container_id: str, llm_answers_table_container_id: str, 
    llm_cot_table_container_id: str, generate_btn_id: str, results_id: str,
):
    """
    Build a basic page layout with a form and edge descriptions table.
    """
    form_section = build_kage_form_section(
        question_input_id, hop_select_id, llm_select_id, kg_select_id, generate_btn_id
    )

    edge_desc_section = build_edge_description_section(description_table_id)
    subgraph_section = build_subgraph_section(subgraph_container_id)
    qa_info_section = build_qa_info_section(qa_info_table_container_id)
    llm_response_section = build_llm_response_section(llm_answers_table_container_id, llm_cot_table_container_id)


    return dbc.Stack(
        [
            html.Div(
                form_section,
            ),
            html.Div(
                llm_response_section
            ),
            html.Div(
                subgraph_section
            ),
            html.Div(
                edge_desc_section
            ),
            html.Div(
                qa_info_section
            ),
        ],
        gap=5,
        style={"padding": 20}
    )
