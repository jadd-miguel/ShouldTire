from dash import dcc, html
from dash.dependencies import Input, Output

def get_layout():
    layout = html.Div(
        children=[
            html.H1(
                "ShouldTire", 
                style={
                    "textAlign": "center",
                }
            ),
            html.H3("Should You Drive On Them?",
                style={
                    "color": "#ddd",
                    "textAlign": "center"
                }
            ),
            html.Div(
                children=[
                    html.Div(
                        children=[
                            html.H4("Metrics"),
                            html.P("Accuracy: 95%"),
                            html.P("Loss: 0.12"),
                        ],
                        style={
                            "padding": "20px",
                            "backgroundColor": "#1e1e1e",
                            "borderRadius": "10px",
                            "width": "45%"
                        }
                    ),
                    html.Div(
                        children=[
                            html.H4("Training Status"),
                            html.P("Epoch: 10/25"),
                            html.P("Status: Running"),
                        ],
                        style={
                            "padding": "20px",
                            "backgroundColor": "#1e1e1e",
                            "borderRadius": "10px",
                            "width": "45%"
                        }
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "padding": "20px"
                }
            )
        ],
        style={
            "fontFamily": "Calibri"
        }
    )
    return layout

# @app.callback(
#     Output("shap_bar", "figure"),
#     Input("feat_nums", "value"),
#     Input("pol_show", "value")
# )
# def update_shap_fig(feat_nums, pol_show):
#     pol_show = pol_show.lower()
#     top_features = [f[0] for f in feature_importance[:feat_nums] if f[2] == pol_show or pol_show == "both"]
#     top_values = [f[1] for f in feature_importance[:feat_nums] if f[2] == pol_show or pol_show == "both"]
#     top_polarities = [f[2] for f in feature_importance[:feat_nums] if f[2] == pol_show or pol_show == "both"]

#     return px.bar(
#         x=top_values[::-1],
#         y=top_features[::-1],
#         color=top_polarities[::-1],
#         labels={
#             "x": "Prediction Magnitude",
#             "y": "Feature"
#         },
#     )

# @app.callback(
#     Output("compare_graph", "figure"),
#     Input("compare_feat", "value"),
#     Input("compare_feat2", "value")
# )
# def update_compare_graph(feat1, feat2):
#     return px.scatter(
#         df_clean,
#         x=feat1,
#         y=feat2
#     )

# @app.callback(
#     Output("count_graph", "figure"),
#     Input("feat", "value"),
#     Input("feat_view", "value")
# )
# def update_count_graph(col, view):

#     counts = df_clean[col].value_counts().reset_index()
#     counts.columns = [col, 'count']

#     if view.lower() == "box plot":
#         return px.box(
#             counts,
#             x='count',
#             log_x=True, 
#             orientation='h'  
#         )

#     return px.histogram(
#         counts,
#         x=col,
#         y='count',
#         log_y=True
#     )

# @app.callback(
#     Output("cat_graph", "figure"),
#     Input("category", "value")
# )
# def update_category_graph(category):

#     top_n = 8
#     counts = df_clean[category].value_counts()
#     top_counts = counts[:top_n]
#     other_count = counts[top_n:].sum()
#     top_counts['other'] = other_count 

#     return px.pie(
#         names=top_counts.index,
#         values=top_counts.values
#     )
