from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import torch
from PIL import Image
from modelling.preprocess import VAL_TS_TRANSFORM
import base64
import io
import torch.nn.functional as F
from util import get_device

def get_layout(app, model):
    with open("assets/output.txt", "r") as f:
        logs = f.read()

    carousel_style = {"width": "150px"}

    layout = html.Div(
        children=[
            dcc.Store(id="store"),
            html.H1(
                "ShouldTire", 
                style={
                    "textAlign": "center",
                }
            ),
            html.Div(
                children=[
                    dcc.Tabs(
                        value="model",
                        children=[
                            dcc.Tab(
                                label="Working Model",
                                value="model",
                                children=[
                                    dcc.Upload(
                                        id="upload-image",
                                        children=html.Button("Upload Image"),
                                        multiple=False,
                                        style={
                                            "margin": "15px",
                                        }
                                    ),
                                    html.Div(
                                        id="verdict",
                                        style={
                                            "fontSize": "32px",
                                        }
                                    ),
                                    html.Div(
                                        children=[
                                            html.Img(
                                                id="selected_img",
                                                src="",
                                                style={
                                                    "width": "500px",
                                                    "height": "500px",
                                                    "padding": "25px 0 25px 0"
                                                }
                                            ),
                                            html.Div([
                                                html.Button(
                                                    html.Img(
                                                        src="/assets/imgs/demo1.jpg",
                                                        style={"width": "150px"}
                                                    ),
                                                    id="img1",
                                                    n_clicks=0,
                                                    style=carousel_style
                                                ),
                                                html.Button(
                                                    html.Img(
                                                        src="/assets/imgs/demo2.jpg",
                                                        style={"width": "150px"}
                                                    ),
                                                    id="img2",
                                                    n_clicks=0,
                                                    style=carousel_style
                                                ),
                                                html.Button(
                                                    html.Img(
                                                        src="/assets/imgs/demo3.jpg",
                                                        style={"width": "150px"}
                                                    ),
                                                    id="img3",
                                                    n_clicks=0,
                                                    style=carousel_style
                                                ),
                                            ]),
                                            html.Button(
                                                "Check Tire",
                                                id="check-tire",
                                                n_clicks=0,
                                                style={
                                                    "fontSize": "48px",
                                                    "margin": "15px",
                                                }
                                            )
                                        ],
                                        style={
                                            "textAlign": "center"
                                        }
                                    )
                                ],
                                style={
                                    "fontSize": "32px"
                                }
                            ),
                            dcc.Tab(
                                label="Pipeline Logs",
                                value="logs",
                                children=[
                                    html.Pre(logs, style={"fontSize": "28px"}),
                                    html.Img(src="/assets/graphs/acc_plot.png", style={"width": "600px"}),
                                    html.Img(src="/assets/graphs/loss_plot.png", style={"width": "600px"}),
                                    html.Img(src="/assets/graphs/conf_matrix.png", style={"width": "600px"})
                                ],
                                style={
                                    "fontSize": "32px",
                                    "textAlign": "center"
                                }
                            )
                        ],
                        style={
                            "border": "2px solid white"
                        }
                    )
                ],
                style={
                    "maxWidth": "1200px",
                    "width": "85%",
                    "margin": "50px auto"
                }
            )
        ],
        style={
            "fontFamily": "Calibri"
        }
    )

    IMG_TYPE = ["FILE_PATH", "ENCODED"]
    @app.callback(
        Output("store", "data"),
        Output("selected_img", "src"),
        Input("img1", "n_clicks"),
        Input("img2", "n_clicks"),
        Input("img3", "n_clicks"),
        Input("upload-image", "contents")
    )
    def select_img(img1, img2, img3, contents):

        if ctx.triggered_id == "upload-image" and contents is not None:
            return {"content": contents, "type": IMG_TYPE[1]}, contents

        selected = None
        match ctx.triggered_id:
            case "img1":
                selected = "/assets/imgs/demo1.jpg"
            case "img2":
                selected = "/assets/imgs/demo2.jpg"
            case "img3":
                selected = "/assets/imgs/demo3.jpg"
            case _:
                selected = ""

        return { "content": selected, "type": IMG_TYPE[0] }, selected

    @app.callback(
        Output("verdict", "children"), 
        Input("check-tire", "n_clicks"),
        State("store", "data")
    )
    def check_tire(_, stored):

        if stored is None:
            return "Upload or select an image of a tire to evaluate"

        tensor = get_tensor(stored)

        with torch.no_grad():
            output = model(tensor.to(get_device()))
        
        probs = F.softmax(output, dim=1)
        defective = round(probs[0, 0].item()*100, 4)
        safe = round(probs[0, 1].item()*100, 4)

        return f"Model thinks it's {defective}% defective and {safe}% safe"

    def get_tensor(stored):

        if stored["type"] == IMG_TYPE[0]:
            image = Image.open(stored["content"][1:])
            tensor = VAL_TS_TRANSFORM(image).unsqueeze(0)
            return tensor
        else:
            _, encoded = stored["content"].split(",")
            data = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            tensor = VAL_TS_TRANSFORM(image).unsqueeze(0)
            return tensor

    return layout
