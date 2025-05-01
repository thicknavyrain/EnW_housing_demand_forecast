import os
import ast
import pandas as pd
import folium
import geopandas as gpd

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
import openai
import plotly.graph_objs as go

# — Load & prep data —
df = pd.read_csv('data/output_mock.csv')
years = sorted({int(c.split('_')[0]) for c in df.columns if c.endswith('housing_price')})
years = [str(y) for y in years]

# gdf_lad = gpd.read_file(
#     'data/Local_Authority_Districts_(April_2023)_Names_and_Codes_in_the_United_Kingdom.shp'
# )
# gdf_lad = gdf_lad.to_crs(epsg=4326)
# gdf_lad['geometry'] = gdf_lad.geometry.simplify(
#     tolerance=0.005, preserve_topology=True
# )

gdf_lad = gpd.read_file("gdf_lad.shp")
gdf_lad = gdf_lad.rename(columns={'LAD24CD':"LAD23CD",
                                  'LAD24NM':"LAD23NM"})

print(gdf_lad.columns)
openai.api_key = os.getenv('OPENAI_API_KEY')

# — Dash setup (light theme + icons) —
external_styles = [
    dbc.themes.FLATLY,
    "https://use.fontawesome.com/releases/v5.8.1/css/all.css"
]
app = dash.Dash(__name__, external_stylesheets=external_styles)
server = app.server

# — Layout: filters (20%), content (80%) —
app.layout = html.Div(style={
    'height': '100vh',
    'display': 'flex',
    'flexDirection': 'column',
    'margin': 0, 'padding': 0
}, children=[

    # Top 20%: Navbar + filters
    html.Div(style={'flex': '0 0 20%', 'display': 'flex', 'flexDirection': 'column'}, children=[
        dbc.NavbarSimple(
            brand="🏠 Housing Forecast",
            color="light", dark=False,
            children=[
                html.Span("UK LA housing trends at a glance", className="navbar-text me-3"),
                dbc.Button("Info", color="secondary", outline=True, size="sm")
            ]
        ),
        html.Div(style={
            'flex': '1',
            'display': 'flex',
            'alignItems': 'center',
            'justifyContent': 'space-around',
            'padding': '0 1rem'
        }, children=[
            # Metric
            html.Div([
                html.Label([html.I(className="fas fa-chart-line me-1"), "Metric"], className='fw-bold'),
                dcc.Dropdown(
                    id='metric-filter',
                    options=[
                        {'label':'Housing Price','value':'housing_price'},
                        {'label':'Affordability','value':'housing_affordability'},
                        {'label':'Net Additional Dwellings','value':'net_additional_dwellings'}
                    ],
                    value='housing_price', clearable=False
                )
            ], style={'width':'22%'}),
            # Experience
            html.Div([
                html.Label([html.I(className="fas fa-user-graduate me-1"), "Experience"], className='fw-bold'),
                dcc.Dropdown(
                    id='exp-filter',
                    options=[
                        {'label':'Beginner','value':'Beginner'},
                        {'label':'Intermediate','value':'Intermediate'},
                        {'label':'Expert','value':'Expert'}
                    ],
                    placeholder='Select level'
                )
            ], style={'width':'22%'}),
            # Local Authority
            html.Div([
                html.Label([html.I(className="fas fa-map-marker-alt me-1"), "Local Authority"], className='fw-bold'),
                dcc.Dropdown(
                    id='la-filter',
                    options=[{'label':n,'value':n} for n in sorted(df['LAD23NM'])],
                    placeholder='Select LA'
                )
            ], style={'width':'22%'}),
            # Year
            html.Div([
                html.Label([html.I(className="fas fa-calendar-alt me-1"), "Year"], className='fw-bold'),
                dcc.Dropdown(
                    id='year-filter',
                    options=[{'label':y,'value':y} for y in years],
                    value=years[0], clearable=False
                )
            ], style={'width':'22%'})
        ])
    ]),

    # Bottom 80%: content columns
    html.Div(style={'flex': '1', 'display': 'flex', 'padding': 0}, children=[

        # Left: map above forecast line
        html.Div(style={'width': '70%', 'display': 'flex', 'flexDirection': 'column'}, children=[
            # Map
            html.Div(style={'flex': '1', 'padding': '0.5rem'}, children=[
                html.Iframe(
                    id='map',
                    style={'width': '100%', 'height': '100%', 'border': 'none'}
                )
            ]),
            # Forecast line
            html.Div(style={'flex': '1', 'padding': '0.5rem', 'paddingTop': 0}, children=[
                dcc.Graph(
                    id='forecast-graph',
                    config={'displayModeBar': False},
                    style={'height': '100%'}
                )
            ])
        ]),

        # Right: summary above bar
        html.Div(style={
            'width': '30%',
            'display': 'flex',
            'flexDirection': 'column',
            'borderLeft': '1px solid #ccc',
            'backgroundColor': '#fafafa'
        }, children=[
            # Summary
            html.Div(style={'flex': '1', 'minHeight': 0, 'overflowY': 'auto', 'padding': '1rem'}, children=[
                html.H5("Summary", className='mt-0'),
                dcc.Markdown(id='llm-summary', style={'whiteSpace': 'pre-wrap'})
            ]),
            # Key Drivers bar
            html.Div(style={'flex': '1', 'minHeight': 0, 'overflow': 'hidden', 'padding': '1rem'}, children=[
                html.H6("Key Drivers", className='mb-1'),
                dcc.Graph(
                    id='importance-bar',
                    config={'displayModeBar': False},
                    style={'height': '100%'}
                )
            ])
        ])
    ])
])

def generate_summary(la, year, metric, exp):
    return "test summary"

@app.callback(
    Output('map',            'srcDoc'),
    Output('forecast-graph', 'figure'),
    Output('importance-bar', 'figure'),
    Output('llm-summary',    'children'),
    Input('la-filter',    'value'),
    Input('year-filter',  'value'),
    Input('metric-filter','value'),
    Input('exp-filter',   'value'),
)
def update(la, year, metric, exp):
    # — Forecast line + CI —
    fig_ts = go.Figure()
    avg    = df[[f"{y}_{metric}" for y in years]].mean()
    lo_nat = df[[f"{y}_{metric}_ci_lower" for y in years]].mean()
    hi_nat = df[[f"{y}_{metric}_ci_upper" for y in years]].mean()

    fig_ts.add_trace(go.Scatter(
        x=years, y=avg, mode='lines',
        name='National Avg', line=dict(shape='spline', width=2)
    ))
    fig_ts.add_trace(go.Scatter(
        x=years, y=hi_nat, mode='lines', line=dict(width=0),
        showlegend=False
    ))
    fig_ts.add_trace(go.Scatter(
        x=years, y=lo_nat, mode='lines',
        fill='tonexty', name='95% CI', line=dict(width=0)
    ))

    if la:
        row   = df[df['LAD23NM']==la].iloc[0]
        hi_la = [row[f"{y}_{metric}_ci_upper"] for y in years]
        lo_la = [row[f"{y}_{metric}_ci_lower"] for y in years]
        fig_ts.add_trace(go.Scatter(
            x=years, y=hi_la, mode='lines', line=dict(width=0), showlegend=False
        ))
        fig_ts.add_trace(go.Scatter(
            x=years, y=lo_la, mode='lines',
            fill='tonexty', name=f'{la} CI', line=dict(width=0)
        ))
        vals = [row[f"{y}_{metric}"] for y in years]
        fig_ts.add_trace(go.Scatter(
            x=years, y=vals, mode='lines+markers',
            name=la, line=dict(shape='spline', width=4)
        ))

    # — Axis formatting & hover —
    yaxis_cfg = {}
    if metric == 'housing_price':
        yaxis_cfg['tickprefix'] = '£'
        yaxis_cfg['hoverformat'] = ',.0f'
    elif metric == 'housing_affordability':
        yaxis_cfg['tickformat'] = '.2f'
        yaxis_cfg['hoverformat'] = '.2f'
    else:
        yaxis_cfg['tickformat'] = ',.0f'
        yaxis_cfg['hoverformat'] = ',.0f'

    fig_ts.update_layout(
        template='plotly_white',
        margin={'l':20,'r':20,'t':30,'b':20},
        showlegend=True,
        hovermode='x unified',
        yaxis=yaxis_cfg
    )

    # — Map —
    map_df = gdf_lad.merge(
        df[['LAD23CD', f"{year}_{metric}"]], on='LAD23CD', how='left'
    )
    m = folium.Map(location=[54.0, -2.0], zoom_start=5, tiles='CartoDB positron')
    folium.Choropleth(
        geo_data=map_df.__geo_interface__,
        data=map_df,
        columns=['LAD23CD', f"{year}_{metric}"],
        key_on='feature.properties.LAD23CD',
        fill_color='YlOrRd', fill_opacity=0.7,
        line_opacity=0.2, highlight=True
    ).add_to(m)
    folium.GeoJson(
        map_df,
        style_function=lambda f: {'fillColor':'transparent','color':'grey','weight':0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=['LAD23NM', f"{year}_{metric}"],
            aliases=['LA','Value'], localize=True
        )
    ).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    map_html = m.get_root().render()

    # — Feature importance bar —
    fig_imp = go.Figure()
    if la and year and metric:
        col = f"{year}_{metric}_feature_importance"
        if col in df.columns:
            imp = ast.literal_eval(df.loc[df['LAD23NM']==la, col].iloc[0])
            items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
            feats_raw, scores = zip(*items)
            feats = [f.replace('_',' ').title() for f in feats_raw]
            fig_imp.add_trace(go.Bar(x=scores, y=feats, orientation='h'))
    fig_imp.update_layout(
        title='Drivers of Metric Value',
        title_x=0.5,
        template='plotly_white',
        margin={'l':80,'r':20,'t':40,'b':20},
        xaxis_title='Importance',
        yaxis={'automargin':True, 'categoryorder':'total descending'},
        showlegend=False
    )

    # — Summary —
    summary = generate_summary(la, year, metric, exp) if all([la, year, metric, exp]) else ""
    return map_html, fig_ts, fig_imp, summary

if __name__ == '__main__':
    app.run_server(debug=True)