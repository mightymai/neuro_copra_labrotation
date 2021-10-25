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
import os
from flask_caching import Cache
import json
import dash_bootstrap_components as dbc



SIMULATE_UPLOAD_DELAY = 0
SIMULATE_WRITE_DELAY = 0
SIMULATE_READ_DELAY = 0
SIMULATE_TRANSFORM_DELAY = 0


app_dir = os.getcwd()
filecache_dir = os.path.join(app_dir, 'cached_files')
if not os.path.exists(filecache_dir):
    os.makedirs(filecache_dir)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'medviewer'


cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    # Note that filesystem cache doesn't work on systems with ephemeral
    # filesystems like Heroku.
    #'CACHE_TYPE': 'filesystem',
    #'CACHE_DIR': 'cache-directory',

    # should be equal to maximum number of users on the app at a single time
    # higher numbers will store more data in the filesystem / redis cache
    'CACHE_THRESHOLD': 100
})
app.config.suppress_callback_exceptions = True


#def serve_layout():
session_id = str(uuid.uuid4())
layout = html.Div(children=[
    html.Div(session_id, id='session-id', style={'display': 'none'}),
        html.Div(id='filecache_marker', style={'display': 'none'}),
        dbc.Row([dbc.Col([html.Div(id='filename_input', style={'margin': '10px', 'lineHeight': '40px', 'size':5})]),
        dbc.Col([dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '90%',
            'height': '55px',
            'lineHeight': '50px',
            'borderWidth': '1.5px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'verticalAlign' : 'baseline',
            'margin': '5px'
        },
        # Allow multiple files to be uploaded
        multiple=False
        )]),
        dbc.Col([html.Button('Add Chart', id='add-chart', n_clicks=0, className='add-chart button1', style={'display':'block', 'float':'right', 'margin-right':'5px'})])]),
        dbc.Row(className='bigdiv', id='dropdown-menus', children=[])
])
    #return layout

app.layout = html.Div([layout])

def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), error_bad_lines=False, warn_bad_lines=False, sep = ';', parse_dates=['Zeitstempel'])
   
    return df

def write_dataframe(session_id, timestamp, df):
    # simulate reading in big data with a delay
    time.sleep(SIMULATE_WRITE_DELAY)
    filename = os.path.join(filecache_dir, 'id_'+str(session_id)+'_time_'+str(timestamp))
    df.to_pickle(filename)

@cache.memoize(timeout=60)
def read_dataframe(session_id, timestamp):
    '''
    Read dataframe from disk, for now just as CSV
    '''
    print('Calling read_dataframe')
    filename = os.path.join(filecache_dir, 'id_'+str(session_id)+'_time_'+str(timestamp))
    df = pd.read_pickle(filename)
    # simulate reading in big data with a delay
    print('** Reading data from disk **')
    time.sleep(SIMULATE_READ_DELAY)
    return df

@app.callback(
    Output('filecache_marker', 'children'),
    [Input('upload-data', 'contents'),
     Input('upload-data', 'filename'),
     Input('upload-data', 'last_modified')],
    [State('session-id', 'children'),
    State('upload-data', 'last_modified')])
def save_file(contents, filename, last_modified, session_id, timestamp):
    # write contents to file
    print('Calling save_file')
    print('New last_modified would be',last_modified)
    if contents is not None:
        print('contents is not None')
        # Simulate large file upload with sleep
        time.sleep(SIMULATE_UPLOAD_DELAY)
        df = parse_data(contents, filename)
        write_dataframe(session_id, timestamp, df)
        print('finished saving file')
        return str(last_modified) # not str()?



@app.callback(
    [Output('dropdown-menus', 'children'),
     Output('filename_input', 'children'),
     Output('dropdown-menus', 'style')],
    [Input('filecache_marker', 'children'),
     State('upload-data', 'last_modified'),
     State('session-id','children'),
     State('upload-data', 'filename'),
     Input("add-chart", "n_clicks"),
     Input('dropdown-menus', 'children'),
     Input('dropdown-menus', 'style'),
     Input({"type": "dynamic-delete", "index": ALL}, "n_clicks")])

def update_options(filecache_marker, timestamp, session_id, filename, n_clicks, children, style, delete):
    input_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]

    if n_clicks and filename is None:
        message = 'Please upload a file first'
        return [], html.B(message), style

    elif 'index' in input_id:
        print('input_id', input_id)
        delete_chart = json.loads(input_id)["index"]
        children = [
            chart
            for chart in children
            if "'index': " + str(delete_chart) not in str(chart)
        ]
        num_children = len(children) 
        if num_children != 0:
            width_val = int(1/num_children * 100) - 6*num_children
            print(width_val)
            new_style={
                    #"width": '{}%'.format(width_val),
                    'width' : '100%',
                    #"display": "inline-block",
                    "outline": "thin lightgrey solid",
                    "padding": 15,
                    #'margin-top' : 10
                    #'margin': 'auto',
                }
        else:
            new_style = {'display':'none'}

        return children, 'current file: '+filename, new_style
    elif filecache_marker is not None:
        try:
            df = read_dataframe(session_id, timestamp)
            patient_numbers = list(set(df.loc[:,'FallNr']))
            lst_pat = [{'label': i, 'value': i} for i in patient_numbers]
    
            categories = list(set(df.loc[:,'Kategorie']))
            lst_cat = [{'label': i, 'value': i} for i in categories]

            num_children = len(children) + 1 # since we are about to add a child
            width_val = int(1/num_children * 100) - 6
            print(width_val)
            new_style={
                #"width": '{}%'.format(width_val),
                'width':'100%',
                #"display": "inline-block",
                "outline": "thin lightgrey solid",
                "padding": '2px',
                #'margin' : 0
            }
            new_child = dbc.Col(className="dynamic-div",
            children=[html.Div([html.Button(
                    "X",
                    className='divbutton button1',
                    id={"type": "dynamic-delete", "index": n_clicks},
                    n_clicks=0,
                    style={"display": "block"}
                    ),
                dcc.Dropdown(
                    id={"type": "dropdown-patients", "index": n_clicks},
                    options=lst_pat,
                    multi=True,             
                    ),
                dcc.Dropdown(
                    id={"type": "dropdown-category", "index": n_clicks},
                    options=lst_cat,
                    multi=False
                    ),
                dcc.Dropdown(
                    id={"type": "dropdown-ident", "index": n_clicks},
                    options=[],
                    multi=True
                    ),
                dcc.Graph(
                    id={"type": "graph", "index": n_clicks},
                    style={'display':'none'}
                    )

            ])])
            children.append(new_child)
            
            return children, 'current file: '+filename, new_style
        except:
            raise ValueError('no data')
    else:
        return [], "no file uploaded yet!", {'display': 'none'}
    

@app.callback(
    [Output({'type': 'graph', 'index': MATCH}, 'figure'),
     Output({'type': 'graph', 'index': MATCH}, 'style')],
    [Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-ident', 'index': MATCH}, 'value'),
     Input('filecache_marker', 'children'),
     State('upload-data', 'last_modified'),
     State('session-id','children'),
     Input('dropdown-menus', 'children')]
)
def build_plot(patient, category, identifier, filecache_marker, timestamp, session_id, children):
    if filecache_marker is not None:
        df = read_dataframe(session_id, timestamp)

        if patient and category and identifier:
            identifier_list = identifier.copy()
            patient.sort()
            if category == 'Labor':
                value_column = 'Laborwert'
                df_list = get_df(df, patient, identifier_list)
                scatter_data = plot_scatter(df_list, value_column)

            if category == 'Vitalwert':
                if 'RR' in identifier_list:
                    value_column = ['Systolic', 'Mean', 'Diastolic']
                    df_list = get_df(df, patient, 'RR')
                    scatter_data = plot_scatter(df_list, value_column)
                    identifier_list.remove('RR')

                if len(identifier_list) != 0: 
                    value_column = 'Wert'
                    df_list = get_df(df, patient, identifier_list)
                    scatter_data = plot_scatter(df_list, value_column)

            if category == 'Bilanz':
                value_column = 'Wert'
                df_list = get_df(df, patient, identifier_list)
                scatter_data = plot_scatter(df_list, value_column)


            fig = go.Figure(data = scatter_data)
            fig.update_layout(transition_duration=500, 
                              margin=dict(l=20, r=5, t=20, b=0), 
                             legend=dict(
                                        yanchor="top",
                                        y=-0.4,
                                        xanchor="left",
                                        x=0.01
                                    ))

            num_children = len(children)# since we are about to add a child
            width_val = int(1/num_children * 100) - 2
            print(width_val, 'graph')

            style={
                "width": '{}vw'.format(width_val),
                #'width': '20%',
                "display": "block",
                "padding": 15,
                #'margin': 'auto',
            }

            return fig, style
        else:
            raise dash.exceptions.PreventUpdate

    else:
        raise dash.exceptions.PreventUpdate

def delta(A, B):
    delta_days = pd.Timedelta(A.iloc[0]['Zeitstempel'] - B.iloc[0]['Zeitstempel']).days
    return delta_days

def get_df(dataframe, patient, identifier_list):
    df_list = []
    if identifier_list == 'RR':
        for p in patient:
            df_temp = dataframe[(dataframe['FallNr']==int(p))&(dataframe['Wertbezeichner']=='RR')]
            df_list.append(df_temp)
    else:
        for p in patient:
            for i in identifier_list:
                df_temp = dataframe[(dataframe['FallNr']==int(p))&(dataframe['Wertbezeichner']==i)]
                df_list.append(df_temp)
    return df_list

def plot_scatter(df_list, value_column):

    first_df = df_list[0]
    first_scatter = get_scatter(first_df, value_column, first=True)
    df_list.remove(first_df)

    if len(df_list) != 0:
        print('in if plot_scatter')
        rest_scatter = get_scatter(first_df, value_column, df_list=df_list, first=False)
        first_scatter.extend(rest_scatter)

    print(first_scatter)
    return first_scatter

def get_scatter(first_df, value_column, df_list=[], first=False):
    scatter_data = []
    if first:
        p = first_df.iloc[0]['FallNr']
        if len(value_column) == 3:
            for v in value_column:
                scatter_data.append(go.Scatter(
                        x=first_df['Zeitstempel'], 
                        y=first_df[v],
                        name='Patient ' + str(p),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel']]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )

        else:
            scatter_data.append(go.Scatter(
                        x=first_df['Zeitstempel'], 
                        y=first_df[value_column],
                        name='Patient ' + str(p),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel']]),
                        text = [first_df.iloc[0]['Wertbezeichner']]*len(first_df),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
    else:
        if len(value_column) == 3:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_temp = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.offsets.Day(delta_temp)
                for v in value_column:                
                    scatter_data.append(go.Scatter(
                            x=A['Zeitstempel'], 
                            y=df_temp[v],
                            name='Patient ' + str(p),
                            customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel']]),
                            text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                            hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                        )
                    )
        else:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_temp = delta(first_df, df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.offsets.Day(delta_temp)
                scatter_data.append(go.Scatter(
                        x=A['Zeitstempel'], 
                        y=df_temp[value_column],
                        name='Patient ' + str(p),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel']]),
                        text = [df_temp.iloc[0]['Wertbezeichner']]*len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                )
    return scatter_data


@app.callback(
    Output({'type': 'dropdown-ident', 'index': MATCH}, 'options'),
    [Input({'type': 'dropdown-patients', 'index': MATCH}, 'value'),
     Input({'type': 'dropdown-category', 'index': MATCH}, 'value'),
     Input('filecache_marker', 'children'),
     State('upload-data', 'last_modified'),
     State('session-id','children')]
)

def get_dropdown_ident(patient, category, filecache_marker, timestamp, session_id):
    if filecache_marker is not None and patient is not None and category is not None:
        df = read_dataframe(session_id, timestamp)
        if patient and category:
            ident_list = []
            for p in patient:
                ident = list(set(df[(df['Kategorie']==category)&(df['FallNr']==p)].loc[:,'Wertbezeichner']))
                # old version
                # ident_list.extend(ident)

                ident_list.append(ident)
            # old version: identifier has to be in *at least one* list
            # ident_list = list(set(ident_list))

            # new version: identifier has to be in *all* lists
            ident_list = list(set.intersection(*map(set, ident_list)))
            lst_ident = [{'label': i, 'value': i} for i in ident_list]
            return lst_ident
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate 


 

if __name__ == '__main__':
    app.run_server(debug=True)