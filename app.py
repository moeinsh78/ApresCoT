import dash
import dash_bootstrap_components as dbc
from dash import Dash, html
from flask import Flask, redirect, request
from services.config import APP_NAME


server = Flask(__name__)

app = Dash(
    name=__name__,
    server=server,
    title=APP_NAME,
    external_stylesheets=[dbc.themes.ZEPHYR, dbc.icons.BOOTSTRAP],
    use_pages=True,
    url_base_pathname="/aprescot/",
    pages_folder="pages"
)


@server.before_request
def index_redirect():
    """
    Redirect root requests to the Examples page
    """
    if request.method == 'GET':
        if request.path == app.config['url_base_pathname']:
            return redirect(f"{app.config['url_base_pathname']}demo")


navbar = dbc.NavbarSimple(
    [
        dbc.NavItem(dbc.NavLink("Full Demo", href="demo")),
        dbc.NavItem(dbc.NavLink("Toy Examples", href="examples")),
        # dbc.NavItem(dbc.NavLink("Knowledge Graph", href="kage")),
    ],
    brand=APP_NAME,
    brand_href="rage",
    color="primary",
    dark=True,
)

app.layout = html.Div(
    [
        navbar,
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True, use_reloader=False)
