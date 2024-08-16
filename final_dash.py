import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('project_dataset.csv')

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load the SVM model
svc_model = joblib.load('svc_model.pkl')

if 'id' in data.columns:
    data.drop('id', axis=1, inplace=True)

# Precompute model statistics
y_pred = svc_model.predict(data.drop('Exited', axis=1))
cm = confusion_matrix(data['Exited'], y_pred)
accuracy = accuracy_score(data['Exited'], y_pred)

# Feature importance calculation (replace with your own logic)
feature_importance_values = np.random.rand(len(data.columns) - 1)

# Generate confusion matrix heatmap
cm_fig = px.imshow(cm, 
                   labels=dict(x="Predicted Label", y="True Label", color="Count"), 
                   x=['0', '1'], y=['0', '1'],
                   title="Confusion Matrix",
                   color_continuous_scale='Blues')

# Generate accuracy score gauge chart
accuracy_fig = go.Figure(go.Indicator(
    mode='gauge+number',
    value=accuracy,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Accuracy Score"},
    gauge={
        'axis': {'range': [None, 1], 'tickvals': [0, 0.5, 1], 'ticktext': ['0', '0.5', '1']},
        'steps': [
            {'range': [0, 0.5], 'color': "red"},
            {'range': [0.5, 1], 'color': "green"}],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': accuracy
        }
    }
))

# Plot feature importance
fig, ax = plt.subplots()
ax.barh(data.columns.drop('Exited'), feature_importance_values)
ax.set_xlabel('Feature Importance')
ax.set_title('Feature Importance Plot')

# Convert plot to base64 encoding
buf = BytesIO()
plt.savefig(buf, format='png')
buf.seek(0)
plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
src = f'data:image/png;base64,{plot_data}'

# Layout for the Home Page
home_page = html.Div([
    html.Div(className='main-content', children=[
        html.Div(className='welcome-banner', children=[
            html.H1("Bank Churn Predictor"),
            html.Div(className='description-box', children=[
                html.P("Welcome to the Bank Churn Predictor Dashboard. This tool helps in predicting the likelihood of a customer leaving the bank based on various factors. Use this dashboard to gain insights into customer churn and make informed decisions to improve customer retention.")
            ]),
            html.A("Go to Basic Stats", href='/basic_stats', className='button')
        ])
    ])
])

# Layout for the Basic Stats Page
basic_stats_page = html.Div([
    html.Div(className='navbar', children=[
        html.H2("Dashboard"),
        html.A("Basic Stats", href='/basic_stats'),
        html.A("Advanced Stats", href='/advanced_stats'),
        html.A("Model Prediction", href='/model_prediction'),
        html.A("Model Stats", href='/model_stats')
    ]),
    html.Div(className='content', children=[
        html.H1("Basic Statistics Dashboard"),
        html.Div(className='plots-container', children=[
            html.Div([
                dcc.Graph(id='scatter-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='scatter-x-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='CreditScore'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='scatter-y-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='EstimatedSalary'),
                html.Label('Enter Threshold Value:'),
                dcc.Input(id='threshold-input', type='number', value=0, step=1)
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='bar-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='bar-x-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='Geography'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='bar-y-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='Exited')
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='density-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='density-x-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='Age'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='density-y-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='EstimatedSalary')
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='line-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='line-x-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='Age'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='line-y-column-dropdown', options=[{'label': col, 'value': col} for col in data.columns], value='CreditScore'),
                html.Label('Adjust marker size:'),
                dcc.Slider(id='line-marker-size-slider', min=2, max=12, step=1, value=8, marks={i: str(i) for i in range(2, 13)}),
                html.Label('Select line color:'),
                dcc.Dropdown(id='line-color-dropdown', options=[{'label': color, 'value': color} for color in px.colors.qualitative.Pastel], value='lightblue')
            ], className='plot-item')
        ])
    ])
])

column_options = [{'label': col, 'value': col} for col in data.columns]

# Layout for the Advanced Stats Page
advanced_stats_page = html.Div([
    html.Div(className='navbar', children=[
        html.H2("Dashboard"),
        html.A("Basic Stats", href='/basic_stats'),
        html.A("Advanced Stats", href='/advanced_stats'),
        html.A("Model Prediction", href='/model_prediction'),
        html.A("Model Stats", href='/model_stats')
    ]),
    html.Div(className='content', children=[
        html.H1("Advanced Statistics Dashboard"),
        html.Div(className='plots-container',children=[
            html.Div([
                dcc.Graph(id='violin-swarm-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='violin-x-column-dropdown', options=column_options, value='Exited'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='violin-y-column-dropdown', options=column_options, value='Age')
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='parallel-coordinate-plot'),
                html.Label('Select categorical variables:'),
                dcc.Dropdown(
                    id='parallel-coordinate-dropdown',
                    options=column_options,
                    value=['Geography', 'Gender'],  # Default value for categorical dropdown (choose as many as needed)
                    multi=True  # Allow multiple selection
                )
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='ridge-plot'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='ridge-x-column-dropdown', options=column_options, value='Age'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='ridge-y-column-dropdown', options=column_options, value='Exited')
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='bubble-chart'),
                html.Label('Select X-axis column:'),
                dcc.Dropdown(id='bubble-x-column-dropdown', options=column_options, value='Age'),
                html.Label('Select Y-axis column:'),
                dcc.Dropdown(id='bubble-y-column-dropdown', options=column_options, value='CreditScore')
            ], className='plot-item'),
            html.Div([
                dcc.Graph(id='sankey-diagram'),
                html.Label('Select column:'),
                dcc.Dropdown(id='sankey-dropdown', options=column_options, value='Geography')
            ], className='plot-item'),
            html.Div([
                html.H1("Choropleth Map of Customers by Geography"),
                dcc.Graph(id='choropleth-map'),
                html.Label('Select Country:'),
                dcc.Dropdown(
                    id='country-dropdown',
                    options=[{'label': country, 'value': country} for country in data['Geography'].unique()],
                    value='France'
                )
            ], className='plot-item')
        ])
    ])
])

# Layout for the Model Prediction Page
model_prediction_page = html.Div([
    html.Div(className='navbar', children=[
        html.H2("Dashboard"),
        html.A("Basic Stats", href='/basic_stats'),
        html.A("Advanced Stats", href='/advanced_stats'),
        html.A("Model Prediction", href='/model_prediction'),
        html.A("Model Stats", href='/model_stats'),
    ]),
    html.Div(className='content', children=[
        html.H1("Model Prediction Dashboard"),
        html.Div(className='input-container', children=[
            html.Label('Credit Score:'),
            dcc.Input(id='credit-score', type='number', placeholder='Enter Credit Score...'),
            html.Label('Geography:'),
            dcc.Dropdown(
                id='geography',
                options=[
                    {'label': 'France', 'value': 0},
                    {'label': 'Germany', 'value': 1},
                    {'label': 'Spain', 'value': 2}
                ],
                placeholder='Select Geography...'
            ),
            html.Label('Gender:'),
            dcc.Dropdown(
                id='gender',
                options=[
                    {'label': 'Male', 'value': 1},
                    {'label': 'Female', 'value': 0}
                ],
                placeholder='Select Gender...'
            ),
            html.Label('Age:'),
            dcc.Input(id='age', type='number', placeholder='Enter Age...'),
            html.Label('Tenure:'),
            dcc.Input(id='tenure', type='number', placeholder='Enter Tenure...'),
            html.Label('Balance:'),
            dcc.Input(id='balance', type='number', placeholder='Enter Balance...'),
            html.Label('Number of Products:'),
            dcc.Input(id='num-products', type='number', placeholder='Enter Number of Products...'),
            html.Label('Has Credit Card:'),
            dcc.Dropdown(
                id='has-credit-card',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Has Credit Card...'
            ),
            html.Label('Is Active Member:'),
            dcc.Dropdown(
                id='is-active-member',
                options=[
                    {'label': 'Yes', 'value': 1},
                    {'label': 'No', 'value': 0}
                ],
                placeholder='Select Is Active Member...'
            ),
            html.Label('Estimated Salary:'),
            dcc.Input(id='estimated-salary', type='number', placeholder='Enter Estimated Salary...'),
            html.Button('Predict', id='predict-button', n_clicks=0),
            html.Div(id='output-prediction')
        ]),
        html.Div(className='model-stats', children=[
            html.Div(className='confusion-matrix', children=[
                dcc.Graph(id='confusion-matrix-plot', figure=cm_fig)
            ]),
            html.Div(className='accuracy-score', children=[
                dcc.Graph(id='accuracy-score-plot', figure=accuracy_fig)
            ]),
            html.Div(className='feature-importance', children=[
                html.Img(id='feature-importance-plot', src=src, style={'width': '100%'})
            ])
        ])
    ])
])


# Callback to update scatter plot
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('scatter-x-column-dropdown', 'value'),
     Input('scatter-y-column-dropdown', 'value'),
     Input('threshold-input', 'value')]
)
def update_scatter_plot(x_axis_column, y_axis_column, threshold_value):
    filtered_data = data[data[y_axis_column] > threshold_value]
    figure = {
        'data': [{'x': filtered_data[x_axis_column], 'y': filtered_data[y_axis_column], 'mode': 'markers', 'name': 'Data Points'}],
        'layout': {
            'title': f'Scatter Plot of {y_axis_column} vs. {x_axis_column}',
            'xaxis': {'title': x_axis_column},
            'yaxis': {'title': y_axis_column},
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
            'hovermode': 'closest'
        }
    }
    return figure

# Callback to update line chart
@app.callback(
    Output('line-plot', 'figure'),
    [Input('line-x-column-dropdown', 'value'),
     Input('line-y-column-dropdown', 'value'),
     Input('line-marker-size-slider', 'value'),
     Input('line-color-dropdown', 'value')]
)
def update_line_chart(x_axis_column, y_axis_column, marker_size, line_color):
    avg_data = data.groupby('Age')[y_axis_column].mean().reset_index()
    figure = {
        'data': [{'x': avg_data['Age'], 'y': avg_data[y_axis_column], 'type': 'line', 'name': y_axis_column, 'marker': {'size': marker_size}, 'line': {'color': line_color}}],
        'layout': {
            'title': f'Average {y_axis_column} by Age',
            'xaxis': {'title': 'Age'},
            'yaxis': {'title': f'Average {y_axis_column}'},
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
            'hovermode': 'closest'
        }
    }
    return figure

# Callback to update bar plot
@app.callback(
    Output('bar-plot', 'figure'),
    [Input('bar-x-column-dropdown', 'value'),
     Input('bar-y-column-dropdown', 'value')]
)
def update_bar_plot(x_axis_column, y_axis_column):
    bar_data = data.groupby(x_axis_column)[y_axis_column].sum().reset_index()
    figure = {
        'data': [{'x': bar_data[x_axis_column], 'y': bar_data[y_axis_column], 'type': 'bar', 'name': y_axis_column}],
        'layout': {
            'title': f'Sum of {y_axis_column} by {x_axis_column}',
            'xaxis': {'title': x_axis_column},
            'yaxis': {'title': f'Sum of {y_axis_column}'},
            'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40},
            'hovermode': 'closest'
        }
    }
    return figure

# Callback to update density plot
@app.callback(
    Output('density-plot', 'figure'),
    [Input('density-x-column-dropdown', 'value'),
     Input('density-y-column-dropdown', 'value')]
)
def update_density_plot(x_axis_column, y_axis_column):
    figure = px.density_heatmap(data, x=x_axis_column, y=y_axis_column, title=f'Density Plot of {y_axis_column} vs. {x_axis_column}')
    return figure


# Callbacks for the violin-swarm plot
@app.callback(
    Output('violin-swarm-plot', 'figure'),
    [
        Input('violin-x-column-dropdown', 'value'),
        Input('violin-y-column-dropdown', 'value')
    ]
)
def update_violin_swarm_plot(x_column, y_column):
    fig = go.Figure()

    fig.add_trace(go.Violin(x=data[x_column][data['Exited'] == 0], y=data[y_column][data['Exited'] == 0],
                             name='Not Exited', box_visible=True, line_color='blue'))
    fig.add_trace(go.Violin(x=data[x_column][data['Exited'] == 1], y=data[y_column][data['Exited'] == 1],
                             name='Exited', box_visible=True, line_color='orange'))

    fig.update_traces(meanline_visible=True)
    fig.update_layout(title=f'{y_column} by {x_column} Violin Plot', xaxis_title=x_column, yaxis_title=y_column)

    return fig

@app.callback(
    Output('parallel-coordinate-plot', 'figure'),
    [Input('parallel-coordinate-dropdown', 'value')]
)
def update_parallel_coordinate_plot(selected_categorical_variables):
    # Create the parallel coordinate plot
    fig = px.parallel_coordinates(data, dimensions=selected_categorical_variables, color='CreditScore',
                                  color_continuous_scale=px.colors.diverging.Tealrose, labels={col: col for col in selected_categorical_variables})
    return fig

# Define callback to update the ridge plot based on selected columns
@app.callback(
    Output('ridge-plot', 'figure'),
    [Input('ridge-x-column-dropdown', 'value'),
     Input('ridge-y-column-dropdown', 'value')]
)
def update_ridge_plot(x_column, y_column):
    fig = px.violin(data, x=x_column, y=y_column, box=True, points="all", title=f'Ridge Plot: {y_column} vs {x_column}')
    return fig

# Define callback to update the bubble chart based on selected columns
@app.callback(
    Output('bubble-chart', 'figure'),
    [Input('bubble-x-column-dropdown', 'value'),
     Input('bubble-y-column-dropdown', 'value')]
)
def update_bubble_chart(x_column, y_column):
    fig = px.scatter(data, x=x_column, y=y_column, size='CreditScore', color='Exited', title=f'Bubble Chart: {y_column} vs {x_column}')
    return fig

# Define callback to update the sankey diagram based on selected column
@app.callback(
    Output('sankey-diagram', 'figure'),
    [Input('sankey-dropdown', 'value')]
)
def update_sankey_diagram(selected_column):
    # Create the sankey diagram
    fig = px.parallel_categories(data, dimensions=[selected_column, 'Exited'], color="Exited",
                                  color_continuous_scale=px.colors.sequential.Inferno,
                                  title=f'Sankey Diagram: {selected_column}')
    return fig

# Define callback to update the choropleth map based on selected column
@app.callback(
    Output('choropleth-map', 'figure'),
    [Input('country-dropdown', 'value')]
)
def update_choropleth_map(selected_country):
    filtered_df = data[data['Geography'] == selected_country]
    geo_data = filtered_df['Geography'].value_counts().reset_index()
    geo_data.columns = ['Country', 'Count']
    
    fig = px.choropleth(geo_data,
                        locations="Country",
                        locationmode='country names',
                        color="Count",
                        hover_name="Country",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title="Number of Customers by Geography")
    
    return fig

# Callback to handle prediction
@app.callback(
    Output('output-prediction', 'children'),
    [
        Input('predict-button', 'n_clicks'),
        Input('credit-score', 'value'),
        Input('geography', 'value'),
        Input('gender', 'value'),
        Input('age', 'value'),
        Input('tenure', 'value'),
        Input('balance', 'value'),
        Input('num-products', 'value'),
        Input('has-credit-card', 'value'),
        Input('is-active-member', 'value'),
        Input('estimated-salary', 'value')
    ]
)
def predict(n_clicks, credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member,estimated_salary):
    if n_clicks is not None and all(v is not None for v in [credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary]):
        # Preprocess inputs if necessary
        input_features = [credit_score, geography, gender, age, tenure, balance, num_products, has_credit_card, is_active_member, estimated_salary]
        # Predict using the SVC model
        prediction = svc_model.predict([input_features])[0]
        return f'Predicted class: {prediction}'
    else:
        return ''

# Define callback to update the model stats
@app.callback(
    [Output('confusion-matrix-plot', 'figure'),
     Output('accuracy-score-plot', 'figure'),
     Output('feature-importance-plot', 'src')],
    [Input('predict-button', 'n_clicks'),
     Input('credit-score', 'value'),
     Input('geography', 'value'),
     Input('gender', 'value'),
     Input('age', 'value'),
     Input('tenure', 'value'),
     Input('balance', 'value'),
     Input('num-products', 'value'),
     Input('has-credit-card', 'value'),
     Input('is-active-member', 'value'),
     Input('estimated-salary', 'value')]
)
def update_model_stats(pathname):
    if pathname == '/model_prediction':
        return cm_fig, accuracy_fig, src
    else:
        return None, None, None

# Initialize the app layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    html.Div(id='dummy-trigger', style={'display': 'none'})  # Dummy trigger for callbacks
])

# Callback to switch between pages
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/basic_stats':
        return basic_stats_page
    elif pathname == '/advanced_stats':
        return advanced_stats_page
    elif pathname == '/model_prediction':
        return model_prediction_page
    else:
        return home_page

if __name__ == '__main__':
    app.run_server(debug=True)