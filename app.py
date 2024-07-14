import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output

# Load the dataset
df = pd.read_csv("C:/Users/yshaa/Downloads/US_Regional_Sales_Data.csv")

# Convert date columns to datetime
df['OrderDate'] = pd.to_datetime(df['OrderDate'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
df['ProcuredDate'] = pd.to_datetime(df['ProcuredDate'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
df['ShipDate'] = pd.to_datetime(df['ShipDate'], format='%d-%m-%Y', errors='coerce', dayfirst=True)
df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], format='%d-%m-%Y', errors='coerce', dayfirst=True)

# Clean and convert 'Unit Price' to numeric
df['Unit Price'] = df['Unit Price'].str.replace(',', '').astype(float)

# Extract additional time features if needed
df['Month'] = df['OrderDate'].dt.to_period('M')
df['Month'] = df['Month'].dt.to_timestamp()
df['Year'] = df['Month'].dt.year

# Start the Dash app
app = dash.Dash(__name__)

# Define app layout
app.layout = html.Div([
    html.H1('Product Demand Forecasting and Price Optimization(ARIMA)'),
    
    dcc.Dropdown(
        id='product-dropdown',
        options=[{'label': product, 'value': product} for product in df['_ProductID'].unique()],
        value=df['_ProductID'].unique()[0],
    ),
    
    dcc.Graph(id='demand-forecast'),
    dcc.Graph(id='price-comparison')
])

# Define callback to update graphs based on product selection
@app.callback(
    [Output('demand-forecast', 'figure'),
     Output('price-comparison', 'figure')],
    [Input('product-dropdown', 'value')]
)

def update_graphs(selected_product):
    print(selected_product)
    # Filter data for the selected product
    product_data = df[df['_ProductID'] == selected_product]
    
    # Compute monthly sales
    monthly_sales = product_data.groupby('Month').agg({'Order Quantity': 'sum'}).reset_index()
    monthly_sales.columns = ['ds', 'y']
    
    # Convert 'ds' to datetime
    monthly_sales['ds'] = pd.to_datetime(monthly_sales['ds'])
    print(monthly_sales)
    
    # Ensure 'ds' is datetime
    if 'ds' not in monthly_sales.columns or monthly_sales.empty:
        raise ValueError(f"Invalid data for Product {selected_product}")
    
    # Fit ARIMA model for demand forecast
    model = ARIMA(monthly_sales['y'], order=(5, 1, 0))  # Example ARIMA order, adjust as needed
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=12)  # Forecast 12 months ahead
    
    # Prepare demand forecast figure
    forecast_index = pd.date_range(start=monthly_sales['ds'].max(), periods=13, freq='ME')[1:]
    fig_demand = px.line(x=forecast_index, y=forecast, title=f'Demand Forecast for Product {selected_product}')
    
    # Compute price comparison
    average_prices = df.groupby(['_ProductID', 'Month'])['Unit Price'].mean().unstack()
    current_month = df['Month'].max()
    current_prices = df[df['Month'] == current_month].groupby('_ProductID')['Unit Price'].mean()
    price_comparison = average_prices.join(current_prices.rename('Current Price'))
    price_comparison['Optimal Price'] = price_comparison.mean(axis=1)
    
    # Prepare price comparison figure
    if selected_product in price_comparison.index and not price_comparison.loc[selected_product].isna().any():
        fig_price = px.bar(price_comparison.loc[selected_product], title=f'Price Comparison for Product {selected_product}')
    else:
        fig_price = {
            'data': [],
            'layout': {
                'title': f'Not enough data to compare prices for Product {selected_product}'
            }
        }
    
    return fig_demand, fig_price

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8888)

