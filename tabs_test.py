import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash import dash_table
from dash.dash_table.Format import Group
import base64
import datetime
import io
import pandas as pd
import plotly.graph_objects as go
import uuid
import time
import numpy as np
import os
from flask_caching import Cache
import json
import dash_bootstrap_components as dbc
import plotly.express as px
from itertools import cycle
from scipy.stats import zscore
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import DateOffset
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput


app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks=True)
app.layout = html.Div(children=[
    dbc.Row([dbc.Col(children=[html.Div(id='filename_input', style={'lineHeight': '40px', 'margin': '10px'}, children=['no file uploaded yet!'])]), #'size':5, 'margin': '10px', 
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
                'verticalAlign' : 'baseline',
                'margin': '5px',
                'margin-bottom':'20px',
                'margin-left': '10%'
            },
            # Allow multiple files to be uploaded
            multiple=False
            )]
    ),
    dbc.Col([html.Button('Add Chart', id='add-chart', n_clicks=0, className='add-chart button1', style={'display':'none'})])], justify='center'), 
    dbc.Row(dcc.Loading(dcc.Store(id='store'), type="dot", color='#651fff'), style={'margin-left':'-35px'}),
    dbc.Row(dcc.Tabs(id="tabs-example-graph", value='tab-1-example-graph', children=[
        dcc.Tab(label='Intrapatient Comparison', value='tab-1-example-graph', children=[]),
        dcc.Tab(label='Interpatient Comparison', value='tab-2-example-graph', children=[]),
    ], colors={ 'border': '#d6d6d6', 'primary': '#651fff', 'background': '#ebe9f0',}), style={'margin-top':'15px', 'padding':-5}),
    dbc.Row(className='bigdiv', id='dropdown-menus', children=[]),
    dbc.Row(className='bigdiv', id='dropdown-menus-2', children=[])
])

@app.callback(
    [ServersideOutput("store", "data"),
     Output('filename_input', 'children'),
     Output('middle', 'children')],
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('filename_input', 'children')])
def save_file(contents, filename, children):
    if contents is not None:
        if '.csv' in filename:
            time.sleep(1)

            # does two main things: 1. reads as pandas dataframe 2. adds z-normalization and differenz of 'Einfuhr'-'Ausfuhr'
            df = parse_data(contents, filename)

            print('finished saving file')

            return df, 'current file: '+filename, html.A(children=[html.Button('Load new data', className='add-chart button1')],href='/', style={'margin-left': '35%', 'padding':3, "text-decoration": "none"}),
        else:
            return [], html.B('The uploaded file has to be a .csv file!'), children

@app.callback(
    [Output('dropdown-menus', 'style'),
     Output('dropdown-menus', 'children'),
     Output('dropdown-menus-2', 'style'),
     Output('dropdown-menus-2', 'children'),
     Output('add-chart', 'children'),
     Output('add-chart', 'className'),
     Output('add-chart','disabled'),
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
    # delete window if 'X' button is clicked
    if tab == 'tab-1-example-graph':
        if 'index' in input_id:
            delete_chart = json.loads(input_id)["index"]
            children = [
                chart
                for chart in children
                if "'index': " + str(delete_chart) not in str(chart)
            ]
            num_children = len(children) 
            if num_children != 0:
                new_style={
                    'width':'100%',
                    "padding": 7.5,
                    'margin' : 'auto',
                    'margin-bottom' : '10px'    
                    }
            else:
                new_style = {'display':'none'}
            print('deleted child')
            return style, children, {'display': 'none'}, [], 'Add chart', 'add-chart button1', False,  {'display':'block', 'float':'right'}

        elif file is not None or file != []:
            try:
                patient_numbers = list(set(file.loc[:,'FallNr']))
                lst_pat = [{'label': i, 'value': i} for i in patient_numbers]
        
                categories = sorted(list(set(file.loc[:,'Kategorie'])))
                lst_cat = [{'label': i, 'value': i} for i in categories]
        
                new_style={
                    'width':'99%'
                    }

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
                                style={'display':'none'}
                                )

                        ])], style={"outline": "thin lightgrey solid", 'padding':10,'margin-bottom': 15, 'margin-right':'15px','margin-left':'25px'})

                children.append(new_child)
                print('added dropdown menus 1')
                return new_style, children, {'display': 'none'}, [], 'Add chart', 'add-chart button1', False,  {'display':'block', 'float':'right'}
            except:
                raise dash.exceptions.PreventUpdate
        else:
            return {'display': 'none'}, [], {'display': 'none'}, [], 'Add chart', 'add-chart button1', True,  {'display':'none'}
    
    elif tab == 'tab-2-example-graph':
        input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
        # delete window if 'X' button is clicked
        if 'index' in input_id:
            delete_chart = json.loads(input_id)["index"]
            children_2 = [
                chart
                for chart in children_2
                if "'index': " + str(delete_chart) not in str(chart)
            ]
            num_children = len(children_2) 
            if num_children != 0:
                new_style={
                    'width':'100%',
                    "padding": 7.5,
                    'margin' : 'auto',
                    'margin-bottom' : '10px'    
                    }
            else:
                new_style = {'display':'none'}
            print('deleted child')
            return {'display': 'none'}, [], new_style, children_2, 'Add chart', 'add-chart button1', False,  {'display':'block', 'float':'right'}

        elif file is not None or file != []:
            try:
                patient_numbers = list(set(file.loc[:,'FallNr']))
                lst_pat = [{'label': i, 'value': i} for i in patient_numbers]
        
                categories = sorted(list(set(file.loc[:,'Kategorie'])))
                lst_cat = [{'label': i, 'value': i} for i in categories]
        
                new_style={
                    'width':'100%',
                    "padding": 7.5,
                    'margin' : 'auto',
                    'margin-bottom' : ''    
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
                                style={'display':'none'}
                                )

                        ])], style={"outline": "thin lightgrey solid", 'padding':10, 'margin-right':5, 'margin-left':5})

                children_2.append(new_child)
                print('added dropdown menus 2')
                return {'display': 'none'}, [], new_style, children_2, 'Add chart', 'add-chart button1', False,  {'display':'block', 'float':'right'}
            except:
                raise dash.exceptions.PreventUpdate
        else:
            return {'display': 'none'}, [], {'display': 'none'}, [], 'Add chart', 'add-chart button1', True,  {'display':'none'}

@app.callback(
    [Output({'type': 'dropdown-ident', 'index': MATCH}, 'options')],
    [Input("store", "data"),
     Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input('tabs-example-graph', 'value')])
def get_dropdown_ident(file, patient, category, tab):
    if tab == 'tab-1-example-graph':
        if file is not None or file != []:
            if patient and category:
                ident_list = []
                ident = list(set(file[(file['Kategorie']==category)&(file['FallNr']==patient)]['Wertbezeichner']))
                # old version
                # ident_list.extend(ident)

                ident_list.append(ident)
                # old version: identifier has to be in *at least one* list
                # ident_list = list(set(ident_list))

                # new version: identifier has to be in *all* lists
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

@app.callback(
    Output({'type': 'dropdown-ident-2', 'index': MATCH}, 'options'),
    [Input("store", "data"),
     Input({'type': 'dropdown-patients-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category-2', 'index': MATCH}, 'value'),
     Input('tabs-example-graph', 'value')])
def get_dropdown_ident(file, patient, category, tab):
    if tab == 'tab-2-example-graph':
        patients = patient
        if file is not None or file != []:
            if patients and category:
                ident_list = []
                for p in patients:
                    ident = list(set(file[(file['Kategorie']==category)&(file['FallNr']==p)].loc[:,'Wertbezeichner']))
                    # old version
                    # ident_list.extend(ident)

                    ident_list.append(ident)
                # old version: identifier has to be in *at least one* list
                # ident_list = list(set(ident_list))

                # new version: identifier has to be in *all* lists
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


@app.callback(
    [Output({'type': 'graph-2', 'index': MATCH}, 'style'),
     Output({'type': 'graph-2', 'index': MATCH}, 'figure')],
    [Input({'type': 'dropdown-patients-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category-2', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-ident-2', 'index': MATCH}, 'value'),
     Input("store", "data"),
     Input('dropdown-menus', 'children'),
     Input('tabs-example-graph', 'value')]
)
def build_plot(patient, category, identifier, file, children, tab):
    if tab == 'tab-2-example-graph':
        if file is not None or file != []:
            unit = '_'
            if patient and category and identifier:
                print('patient', patient, 'category', category, 'identifier', identifier)
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
                        df_list_scatter, _= get_df_2(file, patient, 'RR')
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

            
                fig = go.Figure(data = scatter_data)
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
                elif 'RR' == identifier:
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), 
                        showlegend=True,
                        xaxis_title="date of measurement",
                        yaxis_title='mmHg')
                elif len(scatter_data)==0:
                    raise dash.exceptions.PreventUpdate
                elif len(scatter_data) > 1 and unit != '_':
                    max_val = -1
                    max_scat = None
                    for scat in scatter_data:
                        temp = list(scat.x)
                        diff_temp = temp[0]-temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)


                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename( columns={0 :'dates'}, inplace=True )
                    x_scatter= x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))
                    x_temp2 = list(x_scatter['dates'].apply(lambda n: n.iloc[-1]))

                    day_diff = x_temp1[0] -  x_temp1[-1]
                    tick_labels=[]
                    [tick_labels.append(str(i+1)) for i in range(np.abs(day_diff.days))]
                    base = [list(x_temp1)[0] + datetime.timedelta(days=n) for n in range(np.abs(day_diff.days))]
                    base = [i.replace(hour=12, minute=0, second=0) for i in base]
                    print(tick_labels, 'tick_labels')
                    print('day_diff.days', day_diff.days, 'x_temp1[0]', x_temp1[0], 'x_temp1[-1]', x_temp1[-1])

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
                    print('else build_plot')
                    max_val = -1
                    max_scat = None
                    for scat in scatter_data:
                        temp = list(scat.x)
                        diff_temp = temp[0]-temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)


                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename( columns={0 :'dates'}, inplace=True )
                    x_scatter= x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))
                    x_temp2 = list(x_scatter['dates'].apply(lambda n: n.iloc[-1]))
                    
                    day_diff = x_temp1[0] -  x_temp1[-1] 
                    print('day_diff.days', day_diff.days, 'x_temp1[0]', x_temp1[0], 'x_temp1[-1]', x_temp1[-1])
                    tick_labels=[]
                    [tick_labels.append(str(i+1)) for i in range(np.abs(day_diff.days)+1)]
                    base = [list(x_temp1)[0] + datetime.timedelta(days=n) for n in range(np.abs(day_diff.days)+1)]
                    base = [i.replace(hour=12, minute=0, second=0) for i in base]
                    print(tick_labels, 'tick_labels')
                    


                    fig.update_xaxes(
                        ticktext=tick_labels,
                        tickvals=base,
                    )
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), 
                        showlegend=True,
                        xaxis_title="days since admission",
                        yaxis_title='tba')

                style={
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

@app.callback(
    [Output({'type': 'graph', 'index': MATCH}, 'style'),
     Output({'type': 'graph', 'index': MATCH}, 'figure')],
    [Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-ident', 'index': MATCH}, 'value'),
     Input("store", "data"),
     Input('dropdown-menus', 'children'),
     Input('tabs-example-graph', 'value')]
)
def build_plot(patient, category, identifier, file, children, tab):
    if tab == 'tab-1-example-graph':
        if file is not None or file != []:
            unit = '_'
            if patient and category and identifier:
                identifier_list = identifier                
                if len(identifier_list)==1:
                    
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
                            df_list_scatter, _= get_df(file, patient, 'RR')
                            scatter_data = plot_scatter(df_list_scatter, value_column)
                            identifier_list.remove('RR')

                        if len(identifier_list) != 0: 
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data = plot_scatter(df_list_scatter, value_column)

                    if category == 'Bilanz':
                        print('category==Bilanz')
                        if 'Differenz' in identifier_list:
                            
                            value_column = 'Wert'
                            _, df_list_bar = get_df(file, patient, 'Differenz')
                            scatter_data = plot_scatter(df_list_bar, value_column, bar=True)
                            identifier_list.remove('Differenz')
                            
                        if len(identifier_list) != 0:
                            print('second if in build_plot')
                            value_column = 'Wert'
                            df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)
                            scatter_data = plot_scatter(df_list_scatter, value_column, bilanz=True)

                elif len(identifier_list)>1:
                    scatter_data = []
                    unit = ''
                    if category == 'Labor':
                        value_column = 'Laborwert'
                        df_list_scatter, _ = get_df(file, patient, identifier_list, value_column)

                        scatter_data.extend(plot_scatter(df_list_scatter, value_column))
                        

                    if category == 'Vitalwert':
                        if 'RR' in identifier_list:
                            value_column = ['Systolic', 'Mean', 'Diastolic']
                            df_list_scatter, _= get_df(file, patient, 'RR')
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
                
                fig = go.Figure(data = scatter_data)
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
                elif 'RR' == identifier:
                    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5), 
                        showlegend=True,
                        xaxis_title="date of measurement",
                        yaxis_title='mmHg')
                elif len(scatter_data)==0:
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
                        diff_temp = temp[0]-temp[-1]
                        if np.abs(diff_temp.days) > max_val:
                            max_scat = scat
                            max_val = np.abs(diff_temp.days)


                    x_scatter = pd.DataFrame(max_scat.x)
                    x_scatter.rename( columns={0 :'dates'}, inplace=True )
                    x_scatter= x_scatter.groupby(x_scatter['dates'].dt.date)

                    x_temp1 = list(x_scatter['dates'].apply(lambda n: n.iloc[0]))
                    x_temp2 = list(x_scatter['dates'].apply(lambda n: n.iloc[-1]))

                    day_diff = x_temp1[0] -  x_temp1[-1]
                    tick_labels=[]
                    [tick_labels.append(str(i+1)) for i in range(np.abs(day_diff.days))]
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

                style={
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

###############################################################################################
import base64
import datetime
import io
import pandas as pd
import plotly.graph_objects as go
import uuid
import time
import numpy as np
import os
import json
import plotly.express as px
from itertools import cycle
from scipy.stats import zscore
from dateutil.relativedelta import relativedelta

palette = cycle(px.colors.qualitative.Set1)
palette2 = cycle(px.colors.qualitative.D3)
#palette3 = cycle(px.colors.qualitative.Dark2)


# get 'mean time' of two lists of dates 
def get_date_list(x1,x2):
    date_list = []
    for o,t in zip(x1, x2):
        date_temp = o + (t-o)/2
        date_list.append(date_temp)
    return date_list

# from contents and filename which we get from the uploader of the dashboard, read as pandas dataframe
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), error_bad_lines=False, warn_bad_lines=False, sep = ';', parse_dates=['Zeitstempel'])
    df = dataframe_znorm_differenz(df)
    df.Wertbezeichner = df.Wertbezeichner.str.replace("-", '')
    return df


# adds z-normalization and differenz of 'Einfuhr'-'Ausfuhr'
def dataframe_znorm_differenz(dataframe):
    # apply z-normalization to 'Wert'
    dataframe['Wert_norm'] = None
    dataframe.loc[dataframe.Kategorie=='Vitalwert', 'Wert_norm'] = dataframe[dataframe['Kategorie']=='Vitalwert'].groupby(['Wertbezeichner'])['Wert'].transform(lambda x : zscore(x,ddof=0))

    # apply z-normalization to 'Laborwert'
    dataframe['Laborwert_norm'] = None
    dataframe.loc[dataframe.Kategorie=='Labor', 'Laborwert_norm'] = dataframe[dataframe['Kategorie']=='Labor'].groupby(['Wertbezeichner'])['Laborwert'].transform(lambda x : zscore(x,ddof=0)) 
    
    # get difference of 'Einfuhr' and 'Ausfuhr'
    df_ein_temp = dataframe[dataframe['Wertbezeichner']=='Einfuhr']
    df_ein = df_ein_temp['Wert'].astype(int)
    df_aus_temp = dataframe[dataframe['Wertbezeichner']=='Ausfuhr']
    df_aus = df_aus_temp['Wert'].astype(int)
    df_diff = df_ein.values - df_aus.values
    dataframe['Differenz'] = None
    dataframe.loc[dataframe.Wertbezeichner=='Einfuhr', 'Differenz'] = df_diff
    dataframe.loc[dataframe.Wertbezeichner=='Ausfuhr', 'Differenz'] = -df_diff

    # replace NaN in 'Unit' column with empty string
    dataframe['Unit'] = dataframe['Unit'].fillna('')
    dataframe['Differenz'] = dataframe['Differenz'].astype(float)
    return dataframe


# calculates time difference
def delta(A, B):
    diff = relativedelta(A.iloc[0]['Zeitstempel'], B.iloc[0]['Zeitstempel'])
    return diff.years, diff.months, diff.days, diff.hours, diff.minutes, diff.seconds

def delta_day(A,B):
    diff = relativedelta(A, B)
    return diff.days

def get_df_2(dataframe, patients, identifier, value_column=[]):
    df_list_scatter = []
    df_list_bar = []
    if identifier == 'RR':
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr']==int(p))&(dataframe['Wertbezeichner']=='RR')]
            df_list_scatter.append(df_temp)

    elif identifier == ['Differenz'] or identifier == 'Differenz':
        identifier = 'Differenz'
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr']==int(p))&(dataframe['Kategorie']=='Bilanz')]
            df_list_bar.append(df_temp)

    else:
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr']==int(p))&(dataframe['Wertbezeichner']==identifier)]
            df_list_scatter.append(df_temp)
    return df_list_scatter, df_list_bar

# computes subsets of given dataframe based on selected patient and identifier_list
def get_df(dataframe, patient, identifier_list, value_column=[]):
    df_list_scatter = []
    df_list_bar = []
    if identifier_list == 'RR':
        df_temp = dataframe[(dataframe['FallNr']==int(patient))&(dataframe['Wertbezeichner']=='RR')]
        df_list_scatter.append(df_temp)

    elif identifier_list == ['Differenz'] or 'Differenz' in identifier_list or identifier_list == 'Differenz':
        identifier_list = 'Differenz'
        
        df_temp = dataframe[(dataframe['FallNr']==int(patient))&(dataframe['Kategorie']=='Bilanz')]
        df_list_bar.append(df_temp)

    else:
        for i in identifier_list:
            df_temp = dataframe[(dataframe['FallNr']==int(patient))&(dataframe['Wertbezeichner']==i)]
            df_list_scatter.append(df_temp)
    return df_list_scatter, df_list_bar



# from a list consiting of dataframe (subsets of the main dataframe) this function computes go.Scatter elements via get_scatter and appends them to a list
def plot_scatter_2(df_list, value_column, bar=False, bilanz=False):
    first_scatter = []
    if len(df_list) == 1:
        first_df = df_list[0]
        first_scatter = get_scatter_2(first_df, value_column, first=True, bar=bar)
        df_list.remove(first_df)

    elif len(df_list) > 0:
        first_df = df_list[0]
        first_scatter = get_scatter_2(first_df, value_column, df_list=df_list, first=True, bar=bar) #, bilanz=bilanz)
        df_list.remove(first_df)

        rest_scatter = get_scatter_2(first_df, value_column, df_list=df_list, first=False, bar=bar)#, bilanz=bilanz)
        first_scatter.extend(rest_scatter)

    return first_scatter


# computes go.Scatter elements from dataframes
def get_scatter_2(first_df, value_column, df_list=[], first=False, bar=False, bilanz=False):
    print('get_scatter_2')
    if first and df_list == []:
        palette3 = cycle(px.colors.qualitative.D3)
        scatter_data = []
        patient = first_df.iloc[0]['FallNr']
        # identifier is RR
        if len(value_column) == 3: 
            color_temp = next(palette2)
            for v in value_column:
                scatter_data.append(go.Scatter(
                        marker_color = color_temp,
                        mode='lines',
                        x=first_df['Zeitstempel'].sort_values(), 
                        y=first_df[v],
                        name=v,
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}',
                    )
                )
        # add units
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                #marker_color = color_temp,
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column], first_df['Unit']), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}%{customdata[2]}<br>%{customdata[0]}'
                )
            )
        # identifier = 'Differenz'
        elif bar:
            df_temp_ein = first_df[(first_df.Wertbezeichner=='Einfuhr')]

            x = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()['Zeitstempel']
            x = pd.Series([datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x])
            y_ein = list(df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int))

            df_temp_aus = first_df[(first_df.Wertbezeichner=='Ausfuhr')]
            y_aus = list(df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int))

            y_diff = list(df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int))
            color_1 = next(palette3)

            scatter_data.append(go.Bar(
                marker_color = color_1,
                x=x.sort_values(),
                y=y_ein,
                name='Einfuhr/Ausfuhr',
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_ein), axis=-1),
                hovertext = ['Einfuhr']*len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                )
            )

            scatter_data.append(go.Bar(
                marker_color = color_1,
                x=x,
                y=-np.array(y_aus),
                #name='Ausfuhr',
                showlegend=False,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_aus), axis=-1),
                hovertext = ['Ausfuhr']*len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                )
            )
            scatter_data.append(
                go.Scatter(#marker_color = color_2,
                    line=dict(width=2.5),
                    x=x, # brauchen noch Hälfte der Zahlen
                    y=y_diff,
                    name='Differenz',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_diff), axis=-1),
                    hovertext = ['Differenz']*len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                    )
                )
        elif bilanz:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column].astype(int)), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                )
            )
        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column]), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
        return scatter_data
    scatter_data = []
    palette3 = cycle(px.colors.qualitative.D3)
    if first and df_list != []:

        scatter_data = []
        patient = first_df.iloc[0]['FallNr']

        # identifier is RR
        if len(value_column) == 3: 
            color_temp = next(palette)
            for v in value_column:
                scatter_data.append(go.Scatter(
                        marker_color = color_temp,
                        mode='lines',
                        x=first_df['Zeitstempel'].sort_values(), 
                        y=first_df[v],
                        name='Patient ' + str(patient),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column], # no z-norm here!
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column], first_df['Unit']), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                )
            )
        elif bar:
            color_1 = next(palette3)
            df_temp_ein = first_df[(first_df.Wertbezeichner=='Einfuhr')]

            x = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()['Zeitstempel']
            print(x)
            x = pd.Series([datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x])
            print(x)
            y_ein = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)

            df_temp_aus = first_df[(first_df.Wertbezeichner=='Ausfuhr')]
            y_aus = df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)

            y_diff = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int)
            print('second bar', color_1)
            scatter_data.append(go.Bar(
                marker_color = color_1,
                x=x.sort_values(),
                y=y_ein,
                name='Einfuhr/Ausfuhr Patient '+str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_ein), axis=-1),
                hovertext = ['Einfuhr']*len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                )
            )

            scatter_data.append(go.Bar(
                marker_color = color_1,
                x=x.sort_values(),
                y=-y_aus,
                #name='Ausfuhr',
                showlegend=False,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_aus), axis=-1),
                hovertext = ['Ausfuhr']*len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                )
            )
            scatter_data.append(
                go.Scatter(#marker_color = color_2,
                    line=dict(width=2.5),
                    x=x.sort_values(), # brauchen noch Hälfte der Zahlen
                    y=y_diff,
                    name='Differenz Patient '+str(patient),
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_diff), axis=-1),
                    hovertext = ['Differenz']*len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                    )
                )
           
        elif bilanz:
            #print('value_column', value_column)
            #print('first_df[value_column]', first_df[value_column].notna())
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column]), axis=-1), # .astype(int)
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                )
            )
            
        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column], # no z-norm here!
                name='Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column]), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
    else:
        palette3 = cycle(px.colors.qualitative.D3)
        # identifier is RR
        if len(value_column) == 3:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                color_temp = next(palette)
                for v in value_column:                
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        marker_color = color_temp,
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[v],
                        name='Patient ' + str(p),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
            
        else:
            color_1 = next(palette3)
            for df_temp in df_list:
                patient = df_temp.iloc[0]['FallNr']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)

                if df_temp.iloc[0]['Kategorie'] == 'Labor':
                    scatter_data.append(go.Scatter(
                        mode='markers+lines',
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[value_column], # no z-norm here!
                        name='Patient ' + str(patient),
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]), df_temp[value_column], df_temp['Unit']), axis=-1),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                        )
                    )
                elif bar:
                    color_1 = next(palette3)

                    ####################
                    #first_df = df[(df.FallNr==35054219)&(df.Kategorie=='Bilanz')]
                    #temp = first_df['Zeitstempel'].iloc[0]
                    
                    #df_temp = df[(df.FallNr==35336454)&(df.Kategorie=='Bilanz')]
                    df_temp_ein = df_temp[(df_temp.Wertbezeichner=='Einfuhr')]

                    x_df = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()

                    x = x_df['Zeitstempel']                                                             ############### als hovertext
                    #x_temp = pd.Series([datetime.combine(i,datetime.min.time()).replace(hour=12, minute=0) for i in x])
                    x_timestamp = x.apply(lambda x: pd.Timestamp(str(x) + "T12"))
                    # Timestamp can be defined as: pd.Timestamp("2012-12-15T12")
                    # str(datetime.date) = 2015-10-22
                    # add T12 & convert
                    #x_timestamp

                    y_ein = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)
                    df_temp_aus = df_temp[(df_temp.Wertbezeichner=='Ausfuhr')]
                    y_aus = df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)
                    #print('df_temp_aus')
                    #print(df_temp_aus)

                    y_diff = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int)
                    # x soll timestamp, ber ist dattime.date
                    diff_days = (x_timestamp[0] - first_df['Zeitstempel'].iloc[0])

                    #print(diff_days)
                    x_temp = [(i-diff_days).replace(hour=12, minute=0) for i in x_timestamp]            ############### für plot x-Achse
                    dat =  df_temp_ein.groupby(df_temp_ein['Zeitstempel'].dt.date)
                    x = pd.Series(dat['Zeitstempel'].apply(lambda n: n.iloc[0]))

                    ################
                    """
                    print('third elif')
                    df_temp_ein = df_temp[(df_temp.Wertbezeichner=='Einfuhr')]

                    x_df = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()
                    x = x_df['Zeitstempel']
                    print('x\n', x)

                    y_ein = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)
                    delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df, x_df)
                    print(delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds)
                    x = pd.Series([datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x])

                    x_temp = x + pd.DateOffset(years=delta_years) + pd.DateOffset(months=delta_months) + pd.DateOffset(days=delta_days)# + pd.DateOffset(hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                    x_temp = pd.Series([datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x_temp])
                    print(x_temp.duplicated(keep=False))
                    print('x_temp \n', x_temp)


                    #print('df_temp_ein')
                    print('x_temp \n', x_temp)
                    print('x\n', x)
                    #print(df_temp_ein)
                    
                    df_temp_aus = df_temp[(df_temp.Wertbezeichner=='Ausfuhr')]
                    y_aus = df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)
                    #print('df_temp_aus')
                    #print(df_temp_aus)

                    y_diff = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int)
                    print('x', len(x), 'y_ein', len(y_ein), 'y_aus', len(y_aus), 'y_diff', len(y_diff))
                    """
                    scatter_data.append(
                        go.Bar(
                            marker_color=color_1,
                            x=pd.Series(x_temp).sort_values(), # brauchen noch Hälfte der Zahlen
                            y=y_ein,
                            name='Einfuhr/Ausfuhr Patient ' + str(patient),
                            customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_ein), axis=-1),
                            hovertext = ['Einfuhr']*len(df_temp),
                            hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                            )
                        )

                    scatter_data.append(
                        go.Bar(
                            marker_color=color_1,
                            x=pd.Series(x_temp).sort_values(), # brauchen noch Hälfte der Zahlen
                            y=-y_aus,
                            #name='Ausfuhr Patient ' + str(patient),
                            showlegend=False,
                            customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_aus), axis=-1),
                            hovertext = ['Ausfuhr']*len(df_temp),
                            hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                            )
                        )
                    scatter_data.append(
                        go.Scatter(
                            #marker_color=color_2,
                            line=dict(width=2.5),
                            x=pd.Series(x_temp).sort_values(), # brauchen noch Hälfte der Zahlen
                            y=y_diff,
                            name='Differenz Patient ' + str(patient),
                            customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]), y_diff), axis=-1),
                            hovertext = ['Differenz']*len(df_temp),
                            hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                            )
                        )
                else:
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[value_column], # no z-norm here!
                        name='Patient ' + str(patient),
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]), df_temp[value_column]), axis=-1),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                        )
                    )
    return scatter_data

# from a list consiting of dataframe (subsets of the main dataframe) this function computes go.Scatter elements via get_scatter and appends them to a list
def plot_scatter(df_list, value_column, bar=False, bilanz=False):
    first_scatter = []
    if len(df_list) == 1:
        first_df = df_list[0]
        first_scatter = get_scatter(first_df, value_column, first=True, bar=bar) #  bilanz=bilanz
        df_list.remove(first_df)

    elif len(df_list) > 0:
        first_df = df_list[0]
        first_scatter = get_scatter(first_df, value_column, df_list=df_list, first=True, bar=bar) #, bilanz=bilanz)
        df_list.remove(first_df)
        rest_scatter = get_scatter(first_df, value_column, df_list=df_list, first=False, bar=bar)#, bilanz=bilanz)
        first_scatter.extend(rest_scatter)

    return first_scatter


# computes go.Scatter elements from dataframes
def get_scatter(first_df, value_column, df_list=[], first=False, bar=False, bilanz=False):

    if first and df_list == []:
        scatter_data = []
        patient = first_df.iloc[0]['FallNr']
        wertname = first_df.iloc[0]['Wertbezeichner']

        # identifier is RR
        if len(value_column) == 3: 
            color_temp = next(palette2)
            for v in value_column:
                scatter_data.append(go.Scatter(
                        marker_color = color_temp,
                        mode='lines',
                        x=first_df['Zeitstempel'].sort_values(), 
                        y=first_df[v],
                        name=v,
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
        # add units
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                #marker_color = color_temp,
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column], first_df['Unit']), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}%{customdata[2]}<br>%{customdata[0]}'
                )
            )
        # identifier = 'Differenz'
        elif bar:
            palette3 = cycle(px.colors.qualitative.D3)
            color_1 = next(palette3)
            scatter_data.append(
                go.Bar(
                    marker_color=color_1,
                    x=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values(),  # brauchen noch Hälfte der Zahlen
                    y=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Wert'],
                    name='Einfuhr/Ausfuhr',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values()]), first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz']), axis=-1),
                    hovertext = ['Differenz']*len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                    )
                )

            scatter_data.append(
                go.Bar(
                    marker_color=color_1,
                    x=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values(),  # brauchen noch Hälfte der Zahlen
                    y=-first_df[(first_df.Wertbezeichner=='Ausfuhr')]['Wert'],
                    #name='Ausfuhr',
                    showlegend=False,
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values()]), first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz']), axis=-1),
                    hovertext = ['Differenz']*len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                    )
                )
            scatter_data.append(
                go.Scatter(
                    #marker_color=color_1,
                    line=dict(width=2.5),
                    x=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values(),  # brauchen noch Hälfte der Zahlen
                    y=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz'],
                    name='Differenz',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values()]), first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz']), axis=-1),
                    hovertext = ['Differenz']*len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                    )
                )
        elif bilanz:
            print('first')
            print('value_column', value_column)
            print(first_df[value_column])
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),  
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column].astype(int)), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                )
            )
        else:
            print('first else')
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column]), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
        return scatter_data
    scatter_data = []
    if first and df_list != []:
        scatter_data = []
        patient = first_df.iloc[0]['FallNr']
        wertname = first_df.iloc[0]['Wertbezeichner']

        # identifier is RR
        if len(value_column) == 3: 
            color_temp = next(palette)
            for v in value_column:
                scatter_data.append(go.Scatter(
                        marker_color = color_temp,
                        mode='lines',
                        x=first_df['Zeitstempel'].sort_values(), 
                        y=first_df[v],
                        name=wertname,
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column+'_norm'],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column], first_df['Unit']), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                )
            )
        elif bar:
            scatter_data.append(go.Bar(
                x=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values(),  # brauchen nur Hälfte der Zahlen
                y=first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz'],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df[(first_df.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values()]), first_df[(first_df.Wertbezeichner=='Einfuhr')]['Differenz']), axis=-1),
                hovertext = ['Differenz']*len(first_df),
                hovertemplate='%{hovertext}<br>%{y}<br>%{customdata[0]}'
                )
            )
        elif bilanz:
            print('second')
            print('value_column', value_column)
            print(first_df[value_column])
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),  
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column].astype(int)), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                )
            )
            
        else:
            print('second else')
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(), 
                y=first_df[value_column+'_norm'],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]), first_df[value_column]), axis=-1),
                text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
    else:
        # identifier is RR
        if len(value_column) == 3:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                color_temp = next(palette)
                for v in value_column:                
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        marker_color = color_temp,
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[v],
                        name=wertname,
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
            
        else:
            for df_temp in df_list:
                patient = df_temp.iloc[0]['FallNr']
                wertname = df_temp.iloc[0]['Wertbezeichner']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                
                if df_temp.iloc[0]['Kategorie'] == 'Labor':
                    scatter_data.append(go.Scatter(
                        mode='markers+lines',
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[value_column+'_norm'],
                        name=wertname,
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]), df_temp[value_column], df_temp['Unit']), axis=-1),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                        )
                    )
                elif bar:
                    scatter_data.append(go.Bar(
                        x=df_temp[(df_temp.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values(),  # brauchen noch Hälfte der Zahlen
                        y=df_temp[(df_temp.Wertbezeichner=='Einfuhr')]['Differenz'],
                        name=wertname,
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp[(df_temp.Wertbezeichner=='Einfuhr')]['Zeitstempel'].sort_values()]), df_temp[(df_temp.Wertbezeichner=='Einfuhr')]['Differenz']), axis=-1),
                        hovertext = ['Differenz']*len(df_temp),
                        hovertemplate='%{hovertext}<br>%{y}<br>%{customdata[0]}'
                        )
                    )
                elif bilanz:
                    print('third')
                    print('value_column', value_column)
                    print(first_df[value_column])
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        x=df_temp['Zeitstempel'].sort_values(), 
                        y=df_temp[value_column],
                        name=wertname,
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]), df_temp[value_column]).astype(int), axis=-1),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                        )
                    )
                else:
                    print('third else')
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        x=A['Zeitstempel'].sort_values(), 
                        y=df_temp[value_column+'_norm'],
                        name=wertname,
                        customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]), df_temp[value_column]), axis=-1),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                        )
                    )
    return scatter_data

if __name__ == '__main__':
    app.run_server(debug=True)