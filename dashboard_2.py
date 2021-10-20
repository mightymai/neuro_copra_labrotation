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

app = dash.Dash(__name__)
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


def serve_layout():
    session_id = str(uuid.uuid4())
    layout = html.Div(children=[
    html.Div(session_id, id='session-id', style={'display': 'none'}),
    html.Div(id='filecache_marker', style={'display': 'none'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select File')
        ]),
        style={
            'width': '100%',
            'height': '55px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': 'auto'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='filename_input', style={'margin': '10px'}),
    html.Button('Add Chart', id='add-chart', n_clicks=0, className='add-chart button1', style={'display':'block'}),
    html.Div(className='bigdiv', id='dropdown-menus', children=[])
    ])
    return layout

app.layout = serve_layout

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
     Output('dropdown-menus', 'style'),
     Output({"type": "dynamic-div", "index": ALL}, 'style')],
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
        return [], message, style, []

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
                    "width": '{}%'.format(width_val),
                    "display": "inline-block",
                    "outline": "thin lightgrey solid",
                    "padding": 15,
                    #'margin-top' : 10
                    #'margin': 'auto',
                }
        else:
            new_style = {'display':'none'}

        return children, 'current file: '+filename, new_style, new_style
    elif filecache_marker is not None:
        try:
            df = read_dataframe(session_id, timestamp)
            patient_numbers = list(set(df.loc[:,'FallNr']))
            lst_pat = [{'label': i, 'value': i} for i in patient_numbers]
    
            categories = list(set(df.loc[:,'Kategorie']))
            lst_cat = [{'label': i, 'value': i} for i in categories]

            num_children = len(children) + 1 # since we are about to add a child
            width_val = int(1/num_children * 100) - 2*num_children
            print(width_val)
            new_style={
                "width": '{}%'.format(width_val),
                "display": "inline-block",
                "outline": "thin lightgrey solid",
                "padding": '2%',
                'margin' : 0
            }
            new_child = html.Div(id={"type": "dynamic-div", "index": n_clicks},
            children=[html.Button(
                    "X",
                    className='divbutton button1',
                    id={"type": "dynamic-delete", "index": n_clicks},
                    n_clicks=0,
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

            ])
            children.append(new_child)
            
            return children, 'current file: '+filename, {}, new_style
        except:
            raise ValueError('no data')
    else:
        return [], "no file uploaded yet!", {'display': 'none'}, []
    

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

            if category == 'Labor':
                value_column = 'Laborwert'
                scatter_data = []
                for p in patient:
                    print(p)
                    for i in identifier:
                        print(i)
                        scatter_data.append(go.Scatter(
                                x=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)]['Zeitstempel'], 
                                y=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)][value_column],
                                name='Patient ' + str(p) + ', ' + str(i)

                            )
                        )
                    print(scatter_data)
            if category == 'Vitalwert':
                value_column = 'Wert'
                print(identifier_list)
                scatter_data = []
                if 'RR' in identifier_list:
                    value_column = ['Systolic', 'Mean', 'Diastolic']
                    for p in patient:
                        for v in value_column:
                            scatter_data.append(go.Scatter(
                                    x=df[(df['FallNr']==int(p))&(df['Wertbezeichner']=='RR')]['Zeitstempel'], 
                                    y=df[(df['FallNr']==int(p))&(df['Wertbezeichner']=='RR')][v],
                                    name='Patient ' + str(p) + ', RR'

                                )
                            )
                    identifier_list.remove('RR')
                if len(identifier_list) != 0:  
                    print(identifier_list)
                    for p in patient:
                        for i in identifier_list:
                            scatter_data.append(go.Scatter(
                                    x=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)]['Zeitstempel'], 
                                    y=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)][value_column],
                                    name='Patient ' + str(p) + ', ' + str(i)

                                )
                            )
            if category == 'Bilanz':
                scatter_data = []
                value_column = 'Wert'
                for p in patient:
                    for i in identifier_list:
                        scatter_data.append(go.Scatter(
                                x=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)]['Zeitstempel'], 
                                y=df[(df['FallNr']==int(p))&(df['Wertbezeichner']==i)][value_column],
                                name='Patient ' + str(p) + ', ' + str(i)
                            )
                        )

            fig = go.Figure(data = scatter_data, layout=go.Layout(margin=dict(b=0)))
            fig.update_layout(transition_duration=500)

            num_children = len(children)# since we are about to add a child
            width_val = int(1/num_children * 100) - 4*num_children
            print(width_val, 'graph')

            style={
                "width": '{}vw'.format(width_val),
                "display": "block",
                "padding": 15,
                #'margin': 'auto',
            }

            return fig, style
        else:
            raise dash.exceptions.PreventUpdate

    else:
        raise dash.exceptions.PreventUpdate


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
                ident_list.extend(ident)

            ident_list = list(set(ident_list))
            lst_ident = [{'label': i, 'value': i} for i in ident_list]
            return lst_ident
        else:
            raise dash.exceptions.PreventUpdate
    else:
        raise dash.exceptions.PreventUpdate 


 
if __name__ == '__main__':
    app.run_server(debug=True)