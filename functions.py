import dash
from dash import html
from dash import dcc
from dash.dependencies import ALL, MATCH
import dash_bootstrap_components as dbc
from dash_extensions.enrich import Dash, Output, Input, State, ServersideOutput, FileSystemStore
from scipy import stats
import base64
import datetime
import io
import pandas as pd
import plotly.graph_objects as go
import time
import numpy as np
import os
import json
import plotly.express as px
from itertools import cycle
from scipy.stats import zscore
from dateutil.relativedelta import relativedelta

# color palettes
palette = cycle(px.colors.qualitative.Set1)
palette2 = cycle(px.colors.qualitative.D3)
palette4 = cycle(['#000000', '#a4211c', '#ffff00', '#8855dd'])


def delete_button_activated(children_temp, input_id_temp):
    delete_chart = json.loads(input_id_temp)["index"]
    children_temp = [
        chart
        for chart in children_temp
        if "'index': " + str(delete_chart) not in str(chart)
    ]
    
    return children_temp


def check_norm_dist(df):
    alpha = 1e-3
    vital = df[df['Kategorie'] == 'Vitalwert']['Wertbezeichner'].unique()
    cat_1 = ['Vitalwert'] * len(vital)
    p_list_1 = []
    bool_list_1 = []
    for v in vital:
        data = df[df['Wertbezeichner'] == v]['Wert'].dropna()
        p = np.NaN
        b = np.NaN
        if len(data) > 7:
            k2, p = stats.normaltest(data)
            if p <= alpha:
                b = False
            else:
                b = True
        p_list_1.append(p)
        bool_list_1.append(b)

    labor = df[df['Kategorie'] == 'Labor']['Wertbezeichner'].unique()
    cat_2 = ['Labor'] * len(labor)
    p_list_2 = []
    bool_list_2 = []

    for l in labor:
        data = df[df['Wertbezeichner'] == l]['Laborwert'].dropna()
        p = np.NaN
        b = np.NaN
        if len(data) > 7:
            k2, p = stats.normaltest(data)
            if p <= alpha:
                b = False
            else:
                b = True
        p_list_2.append(p)
        bool_list_2.append(b)
    df_return = pd.DataFrame({
        'Kategorie': np.append(cat_1, cat_2),
        'Wertbezeichner': np.append(vital, labor),
        'p-Wert': np.append(p_list_1, p_list_2),
        'Bool': np.append(bool_list_1, bool_list_2)
    })

    return df_return


# get 'mean time' of two lists of dates
def get_date_list(x1, x2):
    date_list = []
    for o, t in zip(x1, x2):
        date_temp = o + (t - o) / 2
        date_list.append(date_temp)
    return date_list


# from contents and filename which we get from the uploader of the dashboard, read as pandas dataframe
def parse_data(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), error_bad_lines=False, warn_bad_lines=False, sep=';',
                     parse_dates=['Zeitstempel'])
    df_list = check_bilanz([df])
    df = dataframe_znorm_differenz(df_list[0])
    df.Wertbezeichner = df.Wertbezeichner.str.replace("-", '')
    return df


# adds z-normalization and differenz of 'Einfuhr'-'Ausfuhr'
def dataframe_znorm_differenz(dataframe):
    # identifier 'T' for Vitalwert and Labor --> change 'T' to 'Temperatur' for Labor
    dataframe.loc[(dataframe.Wertbezeichner == 'T') & (dataframe.Kategorie == 'Labor'), 'Wertbezeichner'] = 'Temperatur'

    # apply z-normalization to 'Wert'
    dataframe['Wert_norm'] = None
    dataframe.loc[dataframe.Kategorie == 'Vitalwert', 'Wert_norm'] = \
    dataframe[dataframe['Kategorie'] == 'Vitalwert'].groupby(['Wertbezeichner'])['Wert'].transform(
        lambda x: zscore(x, ddof=0))

    # apply z-normalization to 'Laborwert'
    dataframe['Laborwert_norm'] = None
    dataframe.loc[dataframe.Kategorie == 'Labor', 'Laborwert_norm'] = \
    dataframe[dataframe['Kategorie'] == 'Labor'].groupby(['Wertbezeichner'])['Laborwert'].transform(
        lambda x: zscore(x, ddof=0))

    temp = dataframe[dataframe['Wertbezeichner'] == 'RR']['Mean']
    dataframe.loc[dataframe.Wertbezeichner == 'RR', 'Wert_norm'] = (temp - temp.mean()) / temp.std(ddof=0)

    # get difference of 'Einfuhr' and 'Ausfuhr'
    df_ein_temp = dataframe[dataframe['Wertbezeichner'] == 'Einfuhr']
    df_ein = df_ein_temp['Wert'].astype(int)
    df_aus_temp = dataframe[dataframe['Wertbezeichner'] == 'Ausfuhr']
    df_aus = df_aus_temp['Wert'].astype(int)
    df_diff = df_ein.values - df_aus.values
    dataframe['Differenz'] = None
    dataframe.loc[dataframe.Wertbezeichner == 'Einfuhr', 'Differenz'] = df_diff
    dataframe.loc[dataframe.Wertbezeichner == 'Ausfuhr', 'Differenz'] = -df_diff

    # replace NaN in 'Unit' column with empty string
    dataframe['Unit'] = dataframe['Unit'].fillna('')
    dataframe['Differenz'] = dataframe['Differenz'].astype(float)
    return dataframe


# calculates time difference
def delta(A, B):
    diff = relativedelta(A.iloc[0]['Zeitstempel'], B.iloc[0]['Zeitstempel'])
    return diff.years, diff.months, diff.days, diff.hours, diff.minutes, diff.seconds


def delta_day(A, B):
    diff = relativedelta(A, B)
    return diff.days

# computes subsets of given dataframe based on selected patient and identifier_list
def get_df_2(dataframe, patients, identifier, value_column=[]):
    df_list_scatter = []
    df_list_bar = []
    if identifier == 'RR':
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr'] == int(p)) & (dataframe['Wertbezeichner'] == 'RR')]
            df_list_scatter.append(df_temp)

    elif identifier == ['Differenz'] or identifier == 'Differenz':
        identifier = 'Differenz'
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr'] == int(p)) & (dataframe['Kategorie'] == 'Bilanz')]
            df_list_bar.append(df_temp)

    else:
        for p in patients:
            df_temp = dataframe[(dataframe['FallNr'] == int(p)) & (dataframe['Wertbezeichner'] == identifier)]
            df_list_scatter.append(df_temp)
    return df_list_scatter, df_list_bar


# computes subsets of given dataframe based on selected patient and identifier_list
def get_df(dataframe, patient, identifier_list, value_column=[]):
    df_list_scatter = []
    df_list_bar = []
    if identifier_list == 'RR':
        df_temp = dataframe[(dataframe['FallNr'] == int(patient)) & (dataframe['Wertbezeichner'] == 'RR')]
        df_list_scatter.append(df_temp)

    elif identifier_list == ['Differenz'] or 'Differenz' in identifier_list or identifier_list == 'Differenz':
        identifier_list = 'Differenz'

        df_temp = dataframe[(dataframe['FallNr'] == int(patient)) & (dataframe['Kategorie'] == 'Bilanz')]
        df_list_bar.append(df_temp)

    else:
        for i in identifier_list:
            df_temp = dataframe[(dataframe['FallNr'] == int(patient)) & (dataframe['Wertbezeichner'] == i)]
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
        first_scatter = get_scatter_2(first_df, value_column, df_list=df_list, first=True, bar=bar, bilanz=bilanz)
        df_list.remove(first_df)

        rest_scatter = get_scatter_2(first_df, value_column, df_list=df_list, first=False, bar=bar, bilanz=bilanz)
        first_scatter.extend(rest_scatter)

    return first_scatter


# computes go.Scatter elements from dataframes
# NOTE: Goal is to let every plot start at same point!
def get_scatter_2(first_df, value_column, df_list=[], first=False, bar=False, bilanz=False):
    if first and df_list == []:
        palette3 = cycle(px.colors.qualitative.D3)
        palette4 = cycle(['#000000', '#a4211c', '#ffff00', '#8855dd'])
        scatter_data = []
        patient = first_df.iloc[0]['FallNr']
        # identifier is RR
        if len(value_column) == 3:
            color_temp = next(palette2)
            for v in value_column:
                scatter_data.append(go.Scatter(
                    marker_color=color_temp,
                    mode='lines',
                    x=first_df['Zeitstempel'].sort_values(),
                    y=first_df[v],
                    name=v,
                    customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                    text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                    hovertemplate='%{text}<br>%{y}<br>%{customdata}',
                )
                )
        # add units
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                # marker_color = color_temp,
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column], first_df['Unit']), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}%{customdata[2]}<br>%{customdata[0]}'
            )
            )
        # identifier = 'Differenz'
        elif bar:
            df_temp_ein = first_df[(first_df.Wertbezeichner == 'Einfuhr')]

            x = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()['Zeitstempel']
            x = pd.Series(
                [datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x])
            y_ein = list(df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int))

            df_temp_aus = first_df[(first_df.Wertbezeichner == 'Ausfuhr')]
            y_aus = list(df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int))

            y_diff = list(df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int))
            dat = df_temp_ein.groupby(df_temp_ein['Zeitstempel'].dt.date)
            x_oj = pd.Series(dat['Zeitstempel'].apply(lambda n: n.iloc[0]))

            color_1 = next(palette3)
            color_2 = next(palette4)
            scatter_data.append(go.Bar(
                marker_color=color_1,
                x=x.sort_values(),
                y=y_ein,
                name='Einfuhr/Ausfuhr',
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_ein),
                                    axis=-1),
                hovertext=['Einfuhr'] * len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )

            scatter_data.append(go.Bar(
                marker_color=color_1,
                x=x,
                y=-np.array(y_aus),
                showlegend=False,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_aus),
                                    axis=-1),
                hovertext=['Ausfuhr'] * len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )
            scatter_data.append(
                go.Scatter(
                    marker_color=color_2,
                    line=dict(width=2.5),
                    x=x,  # brauchen noch Hälfte der Zahlen
                    y=y_diff,
                    name='Differenz',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_diff),
                                        axis=-1),
                    hovertext=['Differenz'] * len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                )
            )
        elif bilanz:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column].astype(int)), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
            )
            )
        elif first_df.iloc[0]['Kategorie'] == 'Vitalwert':
            time, val, time_show, val_show, _ = cut_timeseries(first_df, value_column)
            scatter_data.append(go.Scatter(
                mode='lines',
                x=time,
                y=val,
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                     val_show), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )

        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column]), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
            )
            )
        return scatter_data
    scatter_data = []
    palette3 = cycle(px.colors.qualitative.D3)
    palette4 = cycle(['#000000', '#a4211c', '#ffff00', '#8855dd'])
    if first and df_list != []:

        scatter_data = []
        patient = first_df.iloc[0]['FallNr']

        # identifier is RR
        if len(value_column) == 3:
            color_temp = next(palette)
            for v in value_column:
                scatter_data.append(go.Scatter(
                    marker_color=color_temp,
                    mode='lines',
                    x=first_df['Zeitstempel'].sort_values(),
                    y=first_df[v],
                    name='Patient ' + str(patient),
                    customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                    text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                    hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                )
                )
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],  # no z-norm here!
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column], first_df['Unit']), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
            )
            )
        elif bar:
            df_temp_ein = first_df[(first_df.Wertbezeichner == 'Einfuhr')]

            x = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()['Zeitstempel']
            x = pd.Series(
                [datetime.datetime.combine(i, datetime.datetime.min.time()).replace(hour=12, minute=0) for i in x])
            y_ein = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)

            df_temp_aus = first_df[(first_df.Wertbezeichner == 'Ausfuhr')]
            y_aus = df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)

            y_diff = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(int)
            dat = df_temp_ein.groupby(df_temp_ein['Zeitstempel'].dt.date)
            x_oj = pd.Series(dat['Zeitstempel'].apply(lambda n: n.iloc[0]))

            color_1 = next(palette3)
            color_2 = next(palette4)
            scatter_data.append(go.Bar(
                marker_color=color_1,
                x=x.sort_values(),
                y=y_ein,
                name='Einfuhr/Ausfuhr Patient ' + str(patient),
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_ein,
                                     [patient] * len(y_ein)), axis=-1),
                hovertext=['Einfuhr'] * len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
            )
            )

            scatter_data.append(go.Bar(
                marker_color=color_1,
                x=x.sort_values(),
                y=-y_aus,
                # name='Ausfuhr',
                showlegend=False,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_aus,
                                     [patient] * len(y_ein)), axis=-1),
                hovertext=['Ausfuhr'] * len(first_df),
                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
            )
            )
            scatter_data.append(
                go.Scatter(
                    marker_color=color_2,
                    line=dict(width=2.5),
                    x=x.sort_values(),  # brauchen noch Hälfte der Zahlen
                    y=y_diff,
                    name='Differenz Patient ' + str(patient),
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x_oj.sort_values()]), y_diff,
                                         [patient] * len(y_ein)), axis=-1),
                    hovertext=['Differenz'] * len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
                )
            )

        elif bilanz:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column]), axis=-1),  # .astype(int)
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
            )
            )
        elif first_df.iloc[0]['Kategorie'] == 'Vitalwert':
            time, val, time_show, val_show, _ = cut_timeseries(first_df, value_column)
            scatter_data.append(go.Scatter(
                mode='lines',
                x=time,
                y=val,
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                     val_show), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )

        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],  # no z-norm here!
                name='Patient ' + str(patient),
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column]), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
            )
            )
    else:
        palette3 = cycle(px.colors.qualitative.D3)
        palette4 = cycle(['#000000', '#a4211c', '#ffff00', '#8855dd'])
        # identifier is RR
        if len(value_column) == 3:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df,
                                                                                                         df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(
                    months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(
                    hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                color_temp = next(palette)
                for v in value_column:
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        marker_color=color_temp,
                        x=A['Zeitstempel'].sort_values(),
                        y=df_temp[v],
                        name='Patient ' + str(p),
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                        text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                    )

        else:
            color_1 = next(palette3)
            color_2 = next(palette4)
            for df_temp in df_list:
                if df_temp.empty:
                    break
                else:
                    patient = df_temp.iloc[0]['FallNr']
                    delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df,
                                                                                                             df_temp)
                    A = df_temp.copy()
                    A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(
                        months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(
                        hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)

                    if df_temp.iloc[0]['Kategorie'] == 'Labor':
                        scatter_data.append(go.Scatter(
                            mode='markers+lines',
                            x=A['Zeitstempel'].sort_values(),
                            y=df_temp[value_column],  # no z-norm here!
                            name='Patient ' + str(patient),
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                                                 df_temp[value_column], df_temp['Unit']), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                        )
                        )
                    elif bar:
                        color_1 = next(palette3)
                        color_2 = next(palette4)

                        df_temp_ein = df_temp[(df_temp.Wertbezeichner == 'Einfuhr')]

                        x_df = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum().reset_index()

                        x = x_df['Zeitstempel']
                        x_timestamp = x.apply(lambda x: pd.Timestamp(str(x) + "T12"))

                        y_ein = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)
                        df_temp_aus = df_temp[(df_temp.Wertbezeichner == 'Ausfuhr')]
                        y_aus = df_temp_aus.groupby([df_temp_aus['Zeitstempel'].dt.date]).sum()['Wert'].astype(int)

                        y_diff = df_temp_ein.groupby([df_temp_ein['Zeitstempel'].dt.date]).sum()['Differenz'].astype(
                            int)
                        # x soll timestamp, ber ist dattime.date
                        diff_days = (x_timestamp[0] - first_df['Zeitstempel'].iloc[0])

                        x_temp = [(i - diff_days).replace(hour=12, minute=0) for i in
                                  x_timestamp]
                        dat = df_temp_ein.groupby(df_temp_ein['Zeitstempel'].dt.date)
                        x = pd.Series(dat['Zeitstempel'].apply(lambda n: n.iloc[0]))

                        scatter_data.append(
                            go.Bar(
                                marker_color=color_1,
                                x=pd.Series(x_temp).sort_values(),  # brauchen noch Hälfte der Zahlen
                                y=y_ein,
                                name='Einfuhr/Ausfuhr Patient ' + str(patient),
                                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]),
                                                     y_ein, [patient] * len(y_ein)), axis=-1),
                                hovertext=['Einfuhr'] * len(df_temp),
                                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
                            )
                        )

                        scatter_data.append(
                            go.Bar(
                                marker_color=color_1,
                                x=pd.Series(x_temp).sort_values(),  # brauchen noch Hälfte der Zahlen
                                y=-y_aus,
                                # name='Ausfuhr Patient ' + str(patient),
                                showlegend=False,
                                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]),
                                                     y_aus, [patient] * len(y_ein)), axis=-1),
                                hovertext=['Ausfuhr'] * len(df_temp),
                                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
                            )
                        )
                        scatter_data.append(
                            go.Scatter(
                                marker_color=color_2,
                                line=dict(width=2.5),
                                x=pd.Series(x_temp).sort_values(),  # brauchen noch Hälfte der Zahlen
                                y=y_diff,
                                name='Differenz Patient ' + str(patient),
                                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in x.sort_values()]),
                                                     y_diff, [patient] * len(y_ein)), axis=-1),
                                hovertext=['Differenz'] * len(df_temp),
                                hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}<br>Patient %{customdata[2]}<extra></extra>'
                            )
                        )
                    elif df_temp.iloc[0]['Kategorie'] == 'Vitalwert':
                        time, val, time_show, val_show, df_A = cut_timeseries(df_temp, value_column,
                                                                              df_A=A['Zeitstempel'].sort_values())
                        scatter_data.append(go.Scatter(
                            mode='lines',
                            x=df_A,
                            y=val,
                            name='Patient ' + str(patient),
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                                 val_show), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                        )
                        )

                    else:
                        scatter_data.append(go.Scatter(
                            mode='lines',
                            x=A['Zeitstempel'].sort_values(),
                            y=df_temp[value_column],  # no z-norm here!
                            name='Patient ' + str(patient),
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                                                 df_temp[value_column]), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                        )
                        )
    return scatter_data


# from a list consiting of dataframe (subsets of the main dataframe) this function computes go.Scatter elements via get_scatter and appends them to a list
def plot_scatter(df_list, value_column, bar=False, bilanz=False):
    first_scatter = []
    if len(df_list) == 1:
        first_df = df_list[0]
        first_scatter = get_scatter(first_df, value_column, first=True, bar=bar)  # bilanz=bilanz
        df_list.remove(first_df)

    elif len(df_list) > 0:
        first_df = df_list[0]
        first_scatter = get_scatter(first_df, value_column, df_list=df_list, first=True, bar=bar, bilanz=bilanz)
        df_list.remove(first_df)
        rest_scatter = get_scatter(first_df, value_column, df_list=df_list, first=False, bar=bar, bilanz=bilanz)
        first_scatter.extend(rest_scatter)

    return first_scatter


# computes go.Scatter elements from dataframes
# NOTE: Goal is to let every plot start at same point!

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
                    marker_color=color_temp,
                    mode='lines',
                    x=first_df['Zeitstempel'].sort_values(),
                    y=first_df[v],
                    name=v,
                    customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                    text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                    hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                )
                )
        # add units
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                # marker_color = color_temp,
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column], first_df['Unit']), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
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
                    x=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Zeitstempel'].sort_values(),
                    # brauchen noch Hälfte der Zahlen
                    y=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Wert'],
                    name='Einfuhr/Ausfuhr',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in
                                                   first_df[(first_df.Wertbezeichner == 'Einfuhr')][
                                                       'Zeitstempel'].sort_values()]),
                                         first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz']), axis=-1),
                    hovertext=['Differenz'] * len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )

            scatter_data.append(
                go.Bar(
                    marker_color=color_1,
                    x=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Zeitstempel'].sort_values(),
                    # brauchen noch Hälfte der Zahlen
                    y=-first_df[(first_df.Wertbezeichner == 'Ausfuhr')]['Wert'],
                    # name='Ausfuhr',
                    showlegend=False,
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in
                                                   first_df[(first_df.Wertbezeichner == 'Einfuhr')][
                                                       'Zeitstempel'].sort_values()]),
                                         first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz']), axis=-1),
                    hovertext=['Differenz'] * len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
            scatter_data.append(
                go.Scatter(
                    line=dict(width=2.5),
                    x=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Zeitstempel'].sort_values(),
                    # brauchen noch Hälfte der Zahlen
                    y=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz'],
                    name='Differenz',
                    customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in
                                                   first_df[(first_df.Wertbezeichner == 'Einfuhr')][
                                                       'Zeitstempel'].sort_values()]),
                                         first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz']), axis=-1),
                    hovertext=['Differenz'] * len(first_df),
                    hovertemplate='%{hovertext}<br>%{customdata[1]}<br>%{customdata[0]}'
                )
            )
        elif bilanz:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column].astype(int)), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
            )
            )
        elif first_df.iloc[0]['Kategorie'] == 'Vitalwert':
            time, val, time_show, val_show, _ = cut_timeseries(first_df, value_column)
            scatter_data.append(go.Scatter(
                mode='lines',
                x=time,
                y=val,
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                     val_show), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )

        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column]), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
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
                    marker_color=color_temp,
                    mode='lines',
                    x=first_df['Zeitstempel'].sort_values(),
                    y=first_df[v],
                    name=wertname,
                    customdata=list([d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                    text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                    hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                )
                )
        elif first_df.iloc[0]['Kategorie'] == 'Labor':
            scatter_data.append(go.Scatter(
                mode='markers+lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column + '_norm'],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column], first_df['Unit']), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
            )
            )
        elif bar:
            prrint('second elif bar')
            scatter_data.append(go.Bar(
                x=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Zeitstempel'].sort_values(),
                # brauchen nur Hälfte der Zahlen
                y=first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz'],
                name=wertname,
                customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in
                                               first_df[(first_df.Wertbezeichner == 'Einfuhr')][
                                                   'Zeitstempel'].sort_values()]),
                                     first_df[(first_df.Wertbezeichner == 'Einfuhr')]['Differenz']), axis=-1),
                hovertext=['Differenz'] * len(first_df),
                hovertemplate='%{hovertext}<br>%{y}<br>%{customdata[0]}'
            )
            )
        elif bilanz:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column].astype(int)), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
            )
            )
        elif first_df.iloc[0]['Kategorie'] == 'Vitalwert':
            time, val, time_show, val_show, _ = cut_timeseries(first_df, value_column, norm='_norm')
            scatter_data.append(go.Scatter(
                mode='lines',
                x=time,
                y=val,
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                     val_show), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
            )
            )


        else:
            scatter_data.append(go.Scatter(
                mode='lines',
                x=first_df['Zeitstempel'].sort_values(),
                y=first_df[value_column + '_norm'],
                name=wertname,
                customdata=np.stack((np.array(
                    [d.strftime('%B %d %Y, %H:%M') for d in first_df['Zeitstempel'].sort_values()]),
                                     first_df[value_column]), axis=-1),
                text=[first_df.iloc[0]['Wertbezeichner']] * len(first_df),
                hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
            )
            )
    else:
        # identifier is RR
        if len(value_column) == 3:
            for df_temp in df_list:
                p = df_temp.iloc[0]['FallNr']
                delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df,
                                                                                                         df_temp)
                A = df_temp.copy()
                A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(
                    months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(
                    hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)
                color_temp = next(palette)
                for v in value_column:
                    scatter_data.append(go.Scatter(
                        mode='lines',
                        marker_color=color_temp,
                        x=A['Zeitstempel'].sort_values(),
                        y=df_temp[v],
                        name=wertname,
                        customdata=list([d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                        text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                        hovertemplate='%{text}<br>%{y}<br>%{customdata}'
                    )
                    )

        else:
            for df_temp in df_list:
                if df_temp.empty:
                    break
                else:
                    patient = df_temp.iloc[0]['FallNr']
                    wertname = df_temp.iloc[0]['Wertbezeichner']
                    delta_years, delta_months, delta_days, delta_hours, delta_minutes, delta_seconds = delta(first_df,
                                                                                                             df_temp)
                    A = df_temp.copy()
                    A['Zeitstempel'] = A['Zeitstempel'] + pd.DateOffset(years=delta_years) + pd.DateOffset(
                        months=delta_months) + pd.DateOffset(days=delta_days) + pd.DateOffset(
                        hours=delta_hours) + pd.DateOffset(minutes=delta_minutes) + pd.DateOffset(seconds=delta_seconds)

                    if df_temp.iloc[0]['Kategorie'] == 'Labor':
                        print('Labor 3')
                        print(value_column)
                        print(df_temp[value_column + '_norm'])
                        scatter_data.append(go.Scatter(
                            mode='markers+lines',
                            x=A['Zeitstempel'].sort_values(),
                            y=df_temp[value_column + '_norm'],
                            name=wertname,
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                                                 df_temp[value_column], df_temp['Unit']), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]} %{customdata[2]}<br>%{customdata[0]}'
                        )
                        )
                    elif bar:
                        print('third elif bar')
                        scatter_data.append(go.Bar(
                            x=df_temp[(df_temp.Wertbezeichner == 'Einfuhr')]['Zeitstempel'].sort_values(),
                            # brauchen noch Hälfte der Zahlen
                            y=df_temp[(df_temp.Wertbezeichner == 'Einfuhr')]['Differenz'],
                            name=wertname,
                            customdata=np.stack((np.array([d.strftime('%B %d %Y, %H:%M') for d in
                                                           df_temp[(df_temp.Wertbezeichner == 'Einfuhr')][
                                                               'Zeitstempel'].sort_values()]),
                                                 df_temp[(df_temp.Wertbezeichner == 'Einfuhr')]['Differenz']), axis=-1),
                            hovertext=['Differenz'] * len(df_temp),
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
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                                                 df_temp[value_column]), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{y}<br>%{customdata[0]}'
                        )
                        )
                    elif first_df.iloc[0]['Kategorie'] == 'Vitalwert':
                        print('vita')
                        time, val, time_show, val_show, df_A = cut_timeseries(df_temp, value_column, norm='_norm',
                                                                              df_A=A['Zeitstempel'].sort_values())
                        scatter_data.append(go.Scatter(
                            mode='lines',
                            x=df_A,
                            y=val,
                            name=wertname,
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in time_show]),
                                                 val_show), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>'
                        )
                        )

                    else:
                        print('third else')
                        print(df_temp[value_column + '_norm'])
                        scatter_data.append(go.Scatter(
                            mode='lines',
                            x=A['Zeitstempel'].sort_values(),
                            y=df_temp[value_column + '_norm'],
                            name=wertname,
                            customdata=np.stack((np.array(
                                [d.strftime('%B %d %Y, %H:%M') for d in df_temp['Zeitstempel'].sort_values()]),
                                                 df_temp[value_column]), axis=-1),
                            text=[df_temp.iloc[0]['Wertbezeichner']] * len(df_temp),
                            hovertemplate='%{text}<br>%{customdata[1]}<br>%{customdata[0]}'
                        )
                        )
    return scatter_data


def cut_timeseries(df_temp, value_column, norm='', max_step=12, df_A=[]):
    lst_index = []

    for i in range(len(list(df_temp['Zeitstempel'])) - 1):
        print(i)
        diff_temp = list(df_temp['Zeitstempel'])[i] - list(df_temp['Zeitstempel'])[i + 1]
        if np.abs(diff_temp.total_seconds() / 3600) >= max_step:
            lst_index.append(i)
    df_time = list(df_temp['Zeitstempel'].sort_values())
    value_column_norm = value_column + norm
    df_val = list(df_temp[value_column_norm])
    df_time_show = list(df_temp['Zeitstempel'].sort_values())

    if norm == '_norm':
        df_val_show = list(df_temp[value_column])

    for i in lst_index:
        df_time = df_time[:i + 1] + [''] + df_time[i + 1:]
        df_val = df_val[:i + 1] + [list(df_val)[i + 1]] + df_val[i + 1:]
        df_time_show = df_time_show[:i + 1] + [list(df_time_show)[i + 1]] + df_time_show[i + 1:]
        lst_index[i + 1:] = [x + 1 for x in lst_index[i + 1:]]
        if df_A is not []:
            df_A = list(df_A[:i + 1]) + [''] + list(df_A[i + 1:])
        if norm == '_norm':
            df_val_show = df_val_show[:i + 1] + [list(df_val_show)[i + 1]] + df_val_show[i + 1:]

    if norm == '':
        df_val_show = df_val

    return df_time, df_val, df_time_show, df_val_show, df_A


# there might be missing values (e.g. 2018)
def check_bilanz(dataframe_list):
    for df_index in range(len(dataframe_list)):
        not_one_list = True
        while not_one_list:
            dataframe = dataframe_list[df_index]
            df_ein_temp = dataframe[dataframe['Wertbezeichner'] == 'Einfuhr']
            df_ein = df_ein_temp['Wert'].astype(int)
            df_aus_temp = dataframe[dataframe['Wertbezeichner'] == 'Ausfuhr']
            df_aus = df_aus_temp['Wert']
            diff_index = [x1 - x2 for (x1, x2) in zip(list(df_ein.index), list(df_aus.index))]
            not_one = [i for i in diff_index if i != 1]
            if not_one != []:
                first_difference = diff_index.index(not_one[0])
                i = list(df_ein.index)[first_difference]
                dataframe = dataframe.drop(i)
                dataframe_list[df_index] = dataframe
            else:
                not_one_list = False
    return dataframe_list
