import os
import ast
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from branca.colormap import linear

import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import openai
import plotly.graph_objs as go

# — Load & prep data —
df = pd.read_csv('data/output.csv')
years = sorted({int(c.split('_')[0]) for c in df.columns if c.endswith('_housing_price')})
years = [str(y) for y in years]

# Load & simplify Local Authority boundaries
gdf_lad = gpd.read_file('gdf_lad.shp')
gdf_lad = gdf_lad.rename(columns={'LAD24CD': 'LAD23CD', 'LAD24NM': 'LAD23NM'})
gdf_lad = gdf_lad.to_crs(epsg=4326)
gdf_lad['geometry'] = gdf_lad.geometry.simplify(tolerance=0.005, preserve_topology=True)

openai.api_key = os.getenv('OPENAI_API_KEY')

# — Dash setup (light theme + icons) —
external_styles = [dbc.themes.FLATLY, 'https://use.fontawesome.com/releases/v5.8.1/css/all.css']
app = dash.Dash(__name__, external_stylesheets=external_styles)
server = app.server

# — Layout: filters (20%), content (80%) —
app.layout = html.Div(style={'height': '100vh', 'display': 'flex', 'flexDirection': 'column', 'margin': 0, 'padding': 0}, children=[
    # Top: Navbar + blurb + filters
    html.Div(style={'flex': '0 0 20%', 'display': 'flex', 'flexDirection': 'column'}, children=[
        dbc.NavbarSimple(
            brand='Predictive data models for housing affordability and supply',
            color='light', dark=False,
            children=[html.Span('Unlocking data-driven decisions for future housing supply', className='navbar-text me-3'), dbc.Button('Info', id='open-info', color='secondary', outline=True, size='sm')]
        ),
        html.Div(style={'padding': '0.5rem 1rem', 'backgroundColor': '#f8f9fa'}, children=[
            html.P("This tool supports data-driven decision-making by integrating population trends, migration patterns, and economic indicators to help councils anticipate and meet housing needs. By improving forecasting accuracy, we can reduce mismatches between housing supply and demand, ease pressure on local services, and guide public investment toward healthier, more resilient communities.")
        ]),
        html.Div(style={'flex': '1', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-around', 'padding': '0 1rem'}, children=[
            # Metric filter
            html.Div([html.Label([html.I(className='fas fa-chart-line me-1'), 'Metric'], className='fw-bold'), dcc.Dropdown(
                id='metric-filter',
                options=[
                    {'label': 'Housing Price', 'value': 'housing_price'},
                    {'label': 'Affordability', 'value': 'housing_affordability'},
                    {'label': 'Net Additional Dwellings', 'value': 'net_additional_dwellings'}
                ],
                value='housing_price', clearable=False
            )], style={'width': '30%'}),
            # Local Authority filter
            html.Div([html.Label([html.I(className='fas fa-map-marker-alt me-1'), 'Local Authority'], className='fw-bold'), dcc.Dropdown(
                id='la-filter',
                options=[{'label': n, 'value': n} for n in sorted(df['LAD23NM'])],
                placeholder='Select LA'
            )], style={'width': '30%'}),
            # Year filter (options set via callback)
            html.Div([html.Label([html.I(className='fas fa-calendar-alt me-1'), 'Year'], className='fw-bold'), dcc.Dropdown(
                id='year-filter',
                value=years[-1], clearable=False
            )], style={'width': '30%'})
        ])
    ]),
    # Bottom: main content
    html.Div(style={'flex': '1', 'display': 'flex', 'padding': 0}, children=[
        # Map + forecast
        html.Div(style={'width': '70%', 'display': 'flex', 'flexDirection': 'column'}, children=[
            html.Div(style={'flex': '1', 'padding': '0.5rem'}, children=[html.Iframe(id='map', style={'width': '100%', 'height': '100%', 'border': 'none'})]),
            html.Div(style={'flex': '1', 'padding': '0.5rem', 'paddingTop': 0}, children=[dcc.Graph(id='forecast-graph', config={'displayModeBar': False}, style={'height': '100%'})])
        ]),
        # Summary + importance
        html.Div(style={'width': '30%', 'display': 'flex', 'flexDirection': 'column', 'borderLeft': '1px solid #ccc', 'backgroundColor': '#fafafa'}, children=[
            html.Div(style={'flex': '1', 'minHeight': 0, 'overflowY': 'auto', 'padding': '1rem'}, children=[html.H5('Summary', className='mt-0'), dcc.Markdown(id='llm-summary', style={'whiteSpace': 'pre-wrap'})]),
            html.Div(style={'flex': '1', 'minHeight': 0, 'overflow': 'hidden', 'padding': '1rem'}, children=[html.H6('Key Drivers', className='mb-1'), dcc.Graph(id='importance-bar', config={'displayModeBar': False}, style={'height': '100%'})])
        ])
    ]),
    # Info modal
    dbc.Modal([
        dbc.ModalHeader('Info'),
        dbc.ModalBody([
            html.H6('Affordability index'), html.P('This is a measure of how affordable homes are in an area. It is calculated using house prices and income data. The higher the figure the less affordable housing.'),
            html.H6('Net additional dwellings'), html.P('This is the number of new homes that have been built in an area, taking into account any losses, for example from demolitions.'),
            html.H6('House prices'), html.P('These are average house prices for the area, taken from sales data.'),
            html.H6('Jobs density'), html.P('This is a measure of the number of jobs in an area. The higher the figure the more jobs there are in the area.')
        ]),
        dbc.ModalFooter(dbc.Button('Close', id='close-info', className='ms-auto', n_clicks=0))
    ], id='info-modal', is_open=False, size='lg')
])

# Summary stub
def generate_summary(la, year, metric):
    return 'test summary'

# Dynamic Year options per Metric
@app.callback(
    Output('year-filter', 'options'),
    Output('year-filter', 'value'),
    Input('metric-filter', 'value')
)
def set_year_options(metric):
    # find all years for this metric
    yrs = sorted({int(col.split('_')[0]) for col in df.columns if col.endswith(f"_{metric}")})
    yrs = [str(y) for y in yrs]
    opts = [{'label': y, 'value': y} for y in yrs]
    val = yrs[-1] if yrs else None
    return opts, val

# Main update callback
@app.callback(
    Output('map', 'srcDoc'),
    Output('forecast-graph', 'figure'),
    Output('importance-bar', 'figure'),
    Output('llm-summary', 'children'),
    Input('la-filter', 'value'),
    Input('year-filter', 'value'),
    Input('metric-filter', 'value')
)
def update(la, year, metric):
    col = f"{year}_{metric}"
    # guard missing data
    if col not in df.columns:
        empty_fig = go.Figure()
        empty_fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, annotations=[{'text': 'No data for that metric/year', 'xref':'paper','yref':'paper','showarrow':False}])
        return "", empty_fig, empty_fig, "No data available for this selection"

    # — Forecast line + CI —
    fig_ts = go.Figure()
    avg = df[[f"{y}_{metric}" for y in years if f"{y}_{metric}" in df.columns]].mean()
    lo_nat = df[[f"{y}_{metric}_ci_lower" for y in years if f"{y}_{metric}_ci_lower" in df.columns]].mean()
    hi_nat = df[[f"{y}_{metric}_ci_upper" for y in years if f"{y}_{metric}_ci_upper" in df.columns]].mean()
    fig_ts.add_trace(go.Scatter(x=years, y=avg, mode='lines', name='National Avg', line=dict(shape='spline', width=2)))
    fig_ts.add_trace(go.Scatter(x=years, y=hi_nat, mode='lines', line=dict(width=0), showlegend=False))
    fig_ts.add_trace(go.Scatter(x=years, y=lo_nat, mode='lines', fill='tonexty', name='95% CI', line=dict(width=0)))
    if la:
        row = df[df['LAD23NM']==la].iloc[0]
        hi_la = [row[f"{y}_{metric}_ci_upper"] for y in years if f"{y}_{metric}_ci_upper" in df.columns]
        lo_la = [row[f"{y}_{metric}_ci_lower"] for y in years if f"{y}_{metric}_ci_lower" in df.columns]
        vals_la = [row[f"{y}_{metric}"] for y in years]
        if hi_la and lo_la:
            fig_ts.add_trace(go.Scatter(x=years, y=hi_la, mode='lines', line=dict(width=0), showlegend=False))
            fig_ts.add_trace(go.Scatter(x=years, y=lo_la, mode='lines', fill='tonexty', name=f'{la} CI', line=dict(width=0)))
        fig_ts.add_trace(go.Scatter(x=years, y=vals_la, mode='lines+markers', name=la, line=dict(shape='spline', width=4)))
    y_cfg = {}
    if metric=='housing_price': y_cfg.update(tickprefix='£', hoverformat=',.0f')
    elif metric=='housing_affordability': y_cfg.update(tickformat='.2f', hoverformat='.2f')
    else: y_cfg.update(tickformat=',.0f', hoverformat=',.0f')
    fig_ts.update_layout(template='plotly_white', margin=dict(l=20,r=20,t=30,b=20), hovermode='x unified', yaxis=y_cfg)

    # — Map —
    map_df = gdf_lad.merge(df[['LAD23CD', col]], on='LAD23CD', how='left')
    m = folium.Map(location=[54.0, -2.0], zoom_start=5, tiles='CartoDB positron')
    vals = map_df[col].dropna()
    bins = np.linspace(vals.min(), vals.max(), 25).tolist()
    choropleth = folium.Choropleth(
        geo_data=map_df,
        data=map_df,
        columns=['LAD23CD', col],
        key_on='feature.properties.LAD23CD',
        fill_color='plasma',
        bins=bins,
        fill_opacity=0.7,
        line_opacity=0.2,
        nan_fill_color='transparent',
        legend_name=f"{year} {metric.replace('_',' ').title()}"
    )
    choropleth.add_to(m)
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(
            fields=['LAD23NM', col],
            aliases=['Local Authority','Value'],
            localize=True
        )
    )
    if la:
        geom = gdf_lad[gdf_lad['LAD23NM']==la].geometry.iloc[0]
        minx, miny, maxx, maxy = geom.bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], max_zoom=8)
    folium.LayerControl(collapsed=True).add_to(m)
    map_html = m.get_root().render()

    # — Feature importance bar —
    fig_imp = go.Figure()
    col_imp = f"{year}_{metric}_feature_importance"
    if la and year and metric and col_imp in df.columns:
        imp_dict = ast.literal_eval(df.loc[df['LAD23NM']==la, col_imp].iloc[0])
        items = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        feats, scores = zip(*items)
        feats = [f.replace('_',' ').title() for f in feats]
        fig_imp.add_trace(go.Bar(x=scores, y=feats, orientation='h'))
    fig_imp.update_layout(template='plotly_white', title='Drivers of Metric Value', title_x=0.5, margin=dict(l=80,r=20,t=40,b=20), xaxis_title='Importance', yaxis=dict(automargin=True, categoryorder='total descending'), showlegend=False)

    summary = generate_summary(la, year, metric) if la and year and metric else ''
    return map_html, fig_ts, fig_imp, summary

# Info modal callback
@app.callback(
    Output('info-modal', 'is_open'),
    Input('open-info', 'n_clicks'), Input('close-info', 'n_clicks'), State('info-modal', 'is_open')
)
def toggle_info(n_open, n_close, is_open):
    if n_open or n_close:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)