import dash
from dash import html
from dash import dcc
from dash.dependencies import ALL, MATCH
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput, FileSystemStore
import datetime
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np
import os
from functions import delete_button_activated, check_norm_dist, parse_data, get_df, get_df_2, plot_scatter, plot_scatter_2

# important to run this on the university network
dash_base_pathname = os.getenv('DASH_BASE_PATHNAME', '/')

app = Dash(url_base_pathname=dash_base_pathname, external_stylesheets=[dbc.themes.BOOTSTRAP],
           prevent_initial_callbacks=True,
           output_defaults={'backend': FileSystemStore(cache_dir='/tmp/file_system_store')})

app.layout = html.Div(children=[
    # Box for file upload
    dbc.Row([dbc.Col(children=[html.Div(id='filename_input', style={'lineHeight': '40px', 'margin': '10px'},
                                        children=['no file uploaded yet!'])]),
             dbc.Col(
                 id='middle',
                 children=[dcc.Upload(
                     id='upload-data',
                     children=html.Div(['Drag and Drop or ',
                                        html.A('Select File')
                                        ]),
                     style={
                         'width': '80%',
                         'height': '55px',
                         'lineHeight': '50px',
                         'borderWidth': '1.5px',
                         'borderStyle': 'dashed',
                         'borderRadius': '5px',
                         'textAlign': 'center',
                         'verticalAlign': 'baseline',
                         'margin': '5px',
                         'margin-bottom': '20px',
                         'margin-left': '10%'
                     },
                     # Allow multiple files to be uploaded
                     multiple=False
                 )]
             ),
             # button for adding a chart (if file was uploaded)
             dbc.Col([html.Button('Add Chart', id='add-chart', n_clicks=0, className='add-chart button1',
                                  style={'display': 'none'})])], justify='center'),
    # loading symbol
    dbc.Row(dcc.Loading(dcc.Store(id='store'), type="dot", color='#651fff'), style={'margin-left': '-35px'}),
    # to store normalized dataframe (see callback below)
    dcc.Store(id='store_norm'),
    # alert if data is normally distributed
    dbc.Alert(id='alert', is_open=False, children=[], duration=2000, color='#ffb11f'),
    # Tabs for two modi
    dbc.Row(dcc.Tabs(id="tabs-example-graph", value='tab-1-intra', children=[
        dcc.Tab(label='Intrapatient Comparison', value='tab-1-intra', children=[]),
        dcc.Tab(label='Interpatient Comparison', value='tab-2-inter', children=[]),
    ], colors={'border': '#d6d6d6', 'primary': '#651fff', 'background': '#ebe9f0', }),
            style={'margin-top': '15px', 'padding': -5, 'margin-bottom': '10px'}),
    # two dropdown menues, but empty at first!
    dbc.Row(className='bigdiv', id='dropdown-menus', children=[]),
    dbc.Row(className='bigdiv', id='dropdown-menus-2', children=[])
])


# after fileupload: store data, compute normalized dataframe, show filename in Dashboard
@app.callback(
    [ServersideOutput("store", "data"),
     ServersideOutput("store_norm", "data"),
     Output('filename_input', 'children'),
     Output('middle', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('filename_input', 'children')])
def save_file(contents, filename, children):
    if contents is not None:
        # make sure it is a csv file
        if '.csv' in filename:
            time.sleep(1)
            # does two main things: 1. reads as pandas dataframe 2. adds z-normalization and differenz of 'Einfuhr'-'Ausfuhr'
            df = parse_data(contents, filename)
            df_norm = check_norm_dist(df)
            return df, df_norm, 'current file: ' + filename, html.A(
                children=[html.Button('Load new data', className='add-chart button1')], href='',
                style={'margin-left': '35%', 'padding': 3, "text-decoration": "none"}),
        else:
            return [], [], html.B('The uploaded file has to be a .csv file!'), children


# show the dropdown menus
@app.callback(
    [Output('dropdown-menus', 'style'),
     Output('dropdown-menus', 'children'),
     Output('dropdown-menus-2', 'style'),
     Output('dropdown-menus-2', 'children'),
     Output('add-chart', 'children'),
     Output('add-chart', 'className'),
     Output('add-chart', 'disabled'),
     Output('add-chart', 'style')],
    [Input("store", "data"),
     Input("add-chart", "n_clicks"),
     Input('dropdown-menus', 'children'),
     Input('dropdown-menus', 'style'),
     Input('dropdown-menus-2', 'children'),
      Input('dropdown-menus-2', 'style'),
     Input({"type": "dynamic-delete", "index": ALL}, "n_clicks"),
     Input({"type": "dynamic-delete-2", "index": ALL}, "n_clicks"),
     Input('tabs-example-graph', 'value')])
def update_options(file, n_clicks, children, style, children_2, style_2, delete, delete_2, tab):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if tab == 'tab-1-intra':
        
        # delete window if 'X' button is clicked
        if 'index' in input_id:
            children = delete_button_activated(children, input_id)
            return style, children, {'display': 'none'}, [], 'Add chart', 'add-chart button1', False, {
                'display': 'block', 'float': 'right'}

        # file was uploaded --> fill dropdown menus dynamically
        elif file is not None or file != []:
            try:
                # unique patient numbers
                patient_numbers = list(set(file.loc[:, 'FallNr']))
                # dropdown menus require special format to be filled
                lst_pat = [{'label': i, 'value': i} for i in patient_numbers]

                # unique categories
                categories = sorted(list(set(file.loc[:, 'Kategorie'])))
                lst_cat = [{'label': i, 'value': i} for i in categories]

                # adapt width of chart
                new_style = {
                    'width': '99%'
                }

                # new chart consisting of 3 dropdown menus, graph (image) and a button to close the chart
                new_child = dbc.Row(className="dynamic-div",
                                    children=[
                                        html.Div([
                                            html.Button(
                                                "X",
                                                className='divbutton button1',
                                                id={"type": "dynamic-delete", "index": n_clicks},
                                                n_clicks=0,
                                                style={"display": "block"}
                                            ),
                                            html.Div(children=[
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-patients", "index": n_clicks},
                                                    options=lst_pat,
                                                    multi=False,
                                                    style={"margin-bottom": 5},
                                                ),
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-category", "index": n_clicks},
                                                    options=lst_cat,
                                                    multi=False,
                                                    style={"margin-bottom": 5},
                                                ),
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-ident", "index": n_clicks},
                                                    options=[],
                                                    multi=True
                                                )
                                            ],
                                                className='dropdownstyle'
                                            ),
                                            dcc.Graph(
                                                id={"type": "graph", "index": n_clicks},
                                                style={'display': 'none'}
                                            )

                                        ])],
                                    style={"outline": "thin lightgrey solid", 'padding': 10, 'margin-bottom': 15,
                                           'margin-right': '15px', 'margin-left': '25px'})

                children.append(new_child)
                return new_style, children, {'display': 'none'}, [], 'Add chart', 'add-chart button1', False, {
                    'display': 'block', 'float': 'right'}
            except:
                raise dash.exceptions.PreventUpdate
        else:
            return {'display': 'none'}, [], {'display': 'none'}, [], 'Add chart', 'add-chart button1', True, {
                'display': 'none'}

    elif tab == 'tab-2-inter':
        input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

        # delete window if 'X' button is clicked
        if 'index' in input_id:
            children_2 = delete_button_activated(children_2, input_id)
            return {'display': 'none'}, [], style_2, children_2, 'Add chart', 'add-chart button1', False, {
                'display': 'block', 'float': 'right'}

        # file was uploaded --> fill dropdown menus dynamically
        elif file is not None or file != []:
            try:
                patient_numbers = list(set(file.loc[:, 'FallNr']))
                lst_pat = [{'label': i, 'value': i} for i in patient_numbers]

                categories = sorted(list(set(file.loc[:, 'Kategorie'])))
                lst_cat = [{'label': i, 'value': i} for i in categories]

                new_style = {
                    'width': '99%'
                }

                new_child = dbc.Row(className="dynamic-div",
                                    children=[
                                        html.Div([
                                            html.Button(
                                                "X",
                                                className='divbutton button1',
                                                id={"type": "dynamic-delete-2", "index": n_clicks},
                                                n_clicks=0,
                                                style={"display": "block"}
                                            ),
                                            html.Div(children=[
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-patients-2", "index": n_clicks},
                                                    options=lst_pat,
                                                    multi=True,
                                                    style={"margin-bottom": 5}
                                                ),
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-category-2", "index": n_clicks},
                                                    options=lst_cat,
                                                    multi=False,
                                                    style={"margin-bottom": 5}
                                                ),
                                                dcc.Dropdown(
                                                    id={"type": "dropdown-ident-2", "index": n_clicks},
                                                    options=[],
                                                    multi=False
                                                )],
                                                className='dropdownstyle'
                                            ),
                                            dcc.Graph(
                                                id={"type": "graph-2", "index": n_clicks},
                                                style={'display': 'none'}
                                            )

                                        ])],
                                    style={"outline": "thin lightgrey solid", 'padding': 10, 'margin-bottom': 15,
                                           'margin-right': '15px', 'margin-left': '25px'})

                children_2.append(new_child)
                return {'display': 'none'}, [], new_style, children_2, 'Add chart', 'add-chart button1', False, {
                    'display': 'block', 'float': 'right'}
            except:
                raise dash.exceptions.PreventUpdate
        else:
            return {'display': 'none'}, [], {'display': 'none'}, [], 'Add chart', 'add-chart button1', True, {
                'display': 'none'}

# for intrapatient modus: fill identifier dropdown menus based on uploaded file 
@app.callback(
    [Output({'type': 'dropdown-ident', 'index': MATCH}, 'options')],
    [Input("store", "data"),
     Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input('tabs-example-graph', 'value')])
def get_dropdown_ident(file, patient, category, tab):
    if tab == 'tab-1-intra':
        if file is not None or file != []:
            if patient and category:
                ident_list = []
                ident = list(set(file[(file['Kategorie'] == category) & (file['FallNr'] == patient)]['Wertbezeichner']))
                ident_list.append(ident)
                # identifier has to be in *all* lists
                ident_list = sorted(list(set.intersection(*map(set, ident_list))))

                # add 'Differenz' subcategory
                if category == 'Bilanz':
                    ident_list.append('Differenz')
                    sorted(ident_list)
                lst_ident = [{'label': i, 'value': i} for i in ident_list]
                return lst_ident
            else:
                raise dash.exceptions.PreventUpdate
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

# for interpatient modus: fill identifier dropdown menus based on uploaded file
@app.callback(
    Output({'type': 'dropdown-ident-2', 'index': MATCH}, 'options'),
    [Input("store", "data"),
     Input({'type': 'dropdown-patients-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category-2', 'index': MATCH}, 'value'),
     Input('tabs-example-graph', 'value')])
def get_dropdown_ident(file, patient, category, tab):
    if tab == 'tab-2-inter':
        patients = patient
        if file is not None or file != []:
            if patients and category:
                ident_list = []
                for p in patients:
                    ident = list(
                        set(file[(file['Kategorie'] == category) & (file['FallNr'] == p)].loc[:, 'Wertbezeichner']))
                    ident_list.append(ident)
               # identifier has to be in *all* lists
                ident_list = sorted(list(set.intersection(*map(set, ident_list))))
                if category == 'Bilanz':
                    ident_list.append('Differenz')
                    sorted(ident_list)
                lst_ident = [{'label': i, 'value': i} for i in ident_list]
                return lst_ident
            else:
                raise dash.exceptions.PreventUpdate
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

# show warning if identifier is not normally distributed for interpatient mode
@app.callback([Output('alert', 'children'),
               Output('alert', 'is_open')],
              [Input({'type': 'dropdown-ident-2', 'index': ALL}, 'value'),
               State('store_norm', 'data')])
def get_alert(identifier, norm_file):
    if identifier == []:
        raise dash.exceptions.PreventUpdate
    elif identifier == None:
        raise dash.exceptions.PreventUpdate
    elif identifier[0] == None:
        raise dash.exceptions.PreventUpdate
    elif 'RR' in identifier:
        raise dash.exceptions.PreventUpdate

    elif identifier[0] in ['Ausfuhr', 'Einfuhr', 'Differenz']:
        raise dash.exceptions.PreventUpdate
    elif list(norm_file[norm_file.Wertbezeichner == identifier[0]]['Bool'])[0] == 0.0:
        alert_message = ['The identifier ', html.B(identifier[0]), ' does not follow a normal distribution.']
        return alert_message, True
    else:
        raise dash.exceptions.PreventUpdate

# show warning if identifier is not normally distributed for intrapatient mode
@app.callback([Output('alert', 'children'),
               Output('alert', 'is_open')],
              [Input({'type': 'dropdown-ident', 'index': ALL}, 'value'),
               State('store_norm', 'data')])
def get_alert(identifier, norm_file):
    if identifier == []:
        raise dash.exceptions.PreventUpdate
    elif identifier == [[]]:
        raise dash.exceptions.PreventUpdate
    elif identifier == None:
        raise dash.exceptions.PreventUpdate
    elif identifier[0] == None:
        raise dash.exceptions.PreventUpdate
    elif 'RR' in identifier:
        raise dash.exceptions.PreventUpdate

    elif identifier[0][0] in ['Ausfuhr', 'Einfuhr', 'Differenz']:
        raise dash.exceptions.PreventUpdate
    not_norm = ''
    for i in identifier[0]:
        if list(norm_file[norm_file.Wertbezeichner == i]['Bool'])[0] == 0.0:
            not_norm = not_norm + i + ', '

    if not_norm != '':
        not_norm = not_norm[:-2]
        alert_message = ['The identifier(s) ', html.B(not_norm), ' do not follow a normal distribution.']
        return alert_message, True
    else:
        raise dash.exceptions.PreventUpdate

# based on dropdown choice: plot data for intrapatient mode
@app.callback(
    [Output({'type': 'graph', 'index': MATCH}, 'style'),
     Output({'type': 'graph', 'index': MATCH}, 'figure')],
    [Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-ident', 'index': MATCH}, 'value'),
     Input("store", "data"),
     Input('tabs-example-graph', 'value')]
)
def build_plot(patient, category, identifier, file, tab):
    if tab == 'tab-1-intra':
        rr = False
        if file is not None or file != []:
            unit = '_'
            if patient and category and identifier:
                identifier_list = identifier
                if len(identifier_list) == 1:

                    if category == 'Labor':
                        value_column = 'Laborwert'
                        df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)

                        if not len(df_list_scatter) == 0:
                            df_unit = df_list_scatter[0]
                        else:
                            raise dash.exceptions.PreventUpdate
                        if not df_unit.empty:
                            unit = list(df_unit.Unit)[0]

                        scatter_data = plot_scatter(df_list_scatter, value_column)

                    if category == 'Vitalwert':
                        if 'RR' in identifier_list:
                            value_column = ['Systolic', 'Mean', 'Diastolic']
                            df_list_scatter, _ = get_df(file, patient, 'RR')
                            scatter_data = plot_scatter(df_list_scatter, value_column)
                            rr = True
                            identifier_list.remove('RR')

                        if len(identifier_list) != 0:
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data = plot_scatter(df_list_scatter, value_column)
                    if category == 'Bilanz':
                        if 'Differenz' in identifier_list:
                            value_column = 'Wert'
                            _, df_list_bar = get_df(file, patient, 'Differenz')
                            scatter_data = plot_scatter(df_list_bar, value_column, bar=True)
                            identifier_list.remove('Differenz')

                        if len(identifier_list) != 0:
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data = plot_scatter(df_list_scatter, value_column, bilanz=True)

                elif len(identifier_list) > 1:
                    scatter_data = []
                    unit = ''
                    if category == 'Labor':
                        value_column = 'Laborwert'
                        df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)

                        scatter_data.extend(plot_scatter(df_list_scatter, value_column))

                    if category == 'Vitalwert':
                        if 'RR' in identifier_list:
                            value_column = ['Systolic', 'Mean', 'Diastolic']
                            df_list_scatter, _ = get_df(file, patient, 'RR')
                            scatter_data.extend(plot_scatter(df_list_scatter, value_column))
                            identifier_list.remove('RR')

                        if len(identifier_list) != 0:
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data.extend(plot_scatter(df_list_scatter, value_column))
                    if category == 'Bilanz':
                        if 'Differenz' in identifier_list:
                            value_column = 'Wert'
                            _, df_list_bar = get_df(file, patient, 'Differenz')
                            scatter_data.extend(plot_scatter(df_list_bar, value_column, bar=True))
                            identifier_list.remove('Differenz')

                        if len(identifier_list) != 0:
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data.extend(plot_scatter(df_list_scatter, value_column, bilanz=True))

                fig = go.Figure(data=scatter_data)
                if len(scatter_data) == 1 and unit != '_':
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title=unit)
                elif len(scatter_data) == 1 and unit == '_':
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title='tba')
                elif rr:
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title='mmHg')
                elif len(scatter_data) == 0:
                    raise dash.exceptions.PreventUpdate
                elif category == 'Bilanz':
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title='ml')
                else:
                    max_val = -1
                    max_scat = None
                    for scat in scatter_data:
                        temp = list(scat.x)
                        diff_temp = temp[0] - temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)

                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename(columns={0: 'dates'}, inplace=True)
                    x_scatter = x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))
                    x_temp2 = list(x_scatter['dates'].apply(lambda n: n.iloc[-1]))

                    day_diff = x_temp1[0] - x_temp1[-1]
                    tick_labels = []
                    [tick_labels.append(str(i + 1)) for i in range(np.abs(day_diff.days))]
                    base = [list(x_temp1)[0] + datetime.timedelta(days=n) for n in range(np.abs(day_diff.days))]
                    base = [i.replace(hour=12, minute=0, second=0) for i in base]

                    fig.update_xaxes(
                        ticktext=tick_labels,
                        tickvals=base,
                    )

                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="day of hospitalization",
                                      yaxis_title='z-normalization')

                style = {
                    'width': '100%',
                    "display": "block",
                    "padding": 15
                }

                fig.update_layout(
                    barmode='relative',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.25),
                    plot_bgcolor='#eaeaf2'
                )

                return style, fig
            else:
                raise dash.exceptions.PreventUpdate

        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate

# based on dropdown choice: plot data for interpatient mode
@app.callback(
    [Output({'type': 'graph-2', 'index': MATCH}, 'style'),
     Output({'type': 'graph-2', 'index': MATCH}, 'figure')],
    [Input({'type': 'dropdown-patients-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-ident-2', 'index': MATCH}, 'value'),
     Input("store", "data"),
     Input('tabs-example-graph', 'value')]
)
def build_plot(patient, category, identifier, file, tab):
    if tab == 'tab-2-inter':
        rr = False
        if file is not None or file != []:
            unit = '_'
            if patient and category and identifier:
                if category == 'Labor':
                    value_column = 'Laborwert'
                    df_list_scatter, _ = get_df_2(file, patient, identifier, value_column)
                    if not len(df_list_scatter) == 0:
                        df_unit = df_list_scatter[0]
                    else:
                        raise dash.exceptions.PreventUpdate
                    if not df_unit.empty:
                        unit = list(df_unit.Unit)[0]
                    scatter_data = plot_scatter_2(df_list_scatter, value_column)
                if category == 'Vitalwert':
                    if 'RR' == identifier:
                        value_column = ['Systolic', 'Mean', 'Diastolic']
                        df_list_scatter, _ = get_df_2(file, patient, 'RR')
                        rr = True
                        scatter_data = plot_scatter_2(df_list_scatter, value_column)
                    else:
                        value_column = 'Wert'
                        df_list_scatter, _ = get_df_2(file, patient, identifier, value_column)
                        scatter_data = plot_scatter_2(df_list_scatter, value_column)

                if category == 'Bilanz':
                    if 'Differenz' == identifier:
                        value_column = 'Wert'
                        _, df_list_bar = get_df_2(file, patient, 'Differenz')
                        scatter_data = plot_scatter_2(df_list_bar, value_column, bar=True)

                    else:
                        value_column = 'Wert'
                        df_list_scatter, _ = get_df_2(file, patient, identifier, value_column)
                        scatter_data = plot_scatter_2(df_list_scatter, value_column, bilanz=True)

                fig = go.Figure(data=scatter_data)
                if len(scatter_data) == 1 and unit != '_':
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title=unit)
                elif len(scatter_data) == 1 and unit == '_':
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title='tba')
                elif rr:
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="date of measurement",
                                      yaxis_title='mmHg')
                elif len(scatter_data) == 0:
                    raise dash.exceptions.PreventUpdate
                elif len(scatter_data) > 1 and unit != '_':
                    max_val = -1
                    max_scat = None
                    for scat in scatter_data:
                        temp = list(scat.x)
                        diff_temp = temp[0] - temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)

                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename(columns={0: 'dates'}, inplace=True)
                    x_scatter = x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))

                    day_diff = x_temp1[0] - x_temp1[-1]
                    tick_labels = []
                    [tick_labels.append(str(i + 1)) for i in range(np.abs(day_diff.days))]
                    base = [list(x_temp1)[0] + datetime.timedelta(days=n) for n in range(np.abs(day_diff.days))]
                    base = [i.replace(hour=12, minute=0, second=0) for i in base]

                    fig.update_xaxes(
                        ticktext=tick_labels,
                        tickvals=base,
                    )
                    fig.update_layout(
                        margin=dict(l=5, r=5, t=5, b=5),
                        showlegend=True,
                        xaxis_title="date of measurement",
                        yaxis_title=unit)
                else:
                    max_val = -1
                    max_scat = None
                    for scat in scatter_data:
                        temp = list(scat.x)
                        diff_temp = temp[0] - temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)

                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename(columns={0: 'dates'}, inplace=True)
                    x_scatter = x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))

                    day_diff = x_temp1[0] - x_temp1[-1]
                    tick_labels = []
                    [tick_labels.append(str(i + 1)) for i in range(np.abs(day_diff.days) + 1)]
                    base = [list(x_temp1)[0] + datetime.timedelta(days=n) for n in range(np.abs(day_diff.days) + 1)]
                    base = [i.replace(hour=12, minute=0, second=0) for i in base]

                    fig.update_xaxes(
                        ticktext=tick_labels,
                        tickvals=base,
                    )
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5),
                                      showlegend=True,
                                      xaxis_title="days since admission",
                                      yaxis_title='tba')

                style = {
                    'width': '100%',
                    "display": "block",
                    "padding": 15
                }
                fig.update_xaxes(
                    ticklabelmode="period")
                fig.update_layout(
                    barmode='relative',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.25),
                    plot_bgcolor='#eaeaf2'
                )

                return style, fig
            else:
                raise dash.exceptions.PreventUpdate

        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)

application = app.server
