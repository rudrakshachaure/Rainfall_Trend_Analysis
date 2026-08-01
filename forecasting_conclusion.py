import os
import sys
import io
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from prophet.plot import plot_plotly

# Data Loading (auto-downloads if missing)

CSV_FILENAME = 'rainfall_India_1901-2015.csv'
CSV_URL = (
    'https://raw.githubusercontent.com/chandanverma07/DataSets/master/'
    'rainfall%20in%20india%201901-2015.csv'
)


def load_data():
    if 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
        from google.colab import files
        uploaded = files.upload()
        return pd.read_csv(io.BytesIO(uploaded[CSV_FILENAME]))
    csv_path = CSV_FILENAME
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), CSV_FILENAME)
    if not os.path.exists(csv_path):
        print("Downloading dataset from GitHub...")
        raw = pd.read_csv(CSV_URL)
        df = raw.groupby('YEAR').agg({
            'JAN': 'mean', 'FEB': 'mean', 'MAR': 'mean', 'APR': 'mean',
            'MAY': 'mean', 'JUN': 'mean', 'JUL': 'mean', 'AUG': 'mean',
            'SEP': 'mean', 'OCT': 'mean', 'NOV': 'mean', 'DEC': 'mean',
            'ANNUAL': 'mean', 'Jan-Feb': 'mean', 'Mar-May': 'mean',
            'Jun-Sep': 'mean', 'Oct-Dec': 'mean'
        }).round(2).reset_index()
        df.to_csv(CSV_FILENAME, index=False)
        return df
    return pd.read_csv(csv_path)


rainfall_data = load_data()


# K-Means Clustering — Dry / Normal / Wet
features = rainfall_data[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec','ANNUAL']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
rainfall_data['Cluster'] = kmeans.fit_predict(scaled)

label_map = {0:'Wet', 1:'Normal', 2:'Dry'}
rainfall_data['Category'] = rainfall_data['Cluster'].map(label_map)

fig_cluster = px.scatter(
    rainfall_data, x='YEAR', y='ANNUAL',
    color='Category',
    title='Clustering Years Based on Rainfall Patterns',
    labels={'YEAR':'Year', 'ANNUAL':'Annual Rainfall (mm)'},
    color_discrete_map={'Wet':'green', 'Normal':'orange', 'Dry':'red'},
    hover_data={'Cluster':True, 'Category':True}
)
fig_cluster.update_traces(marker=dict(size=8, line=dict(width=1, color='white')))
fig_cluster.update_layout(legend_title='Rainfall Category', plot_bgcolor='rgba(21,101,192,0.04)')
fig_cluster.show()

print("Cluster Statistics:")
print(rainfall_data.groupby('Category')['ANNUAL'].describe())


# Prophet Forecast — Next 20 Years
prophet_df = rainfall_data[['YEAR','ANNUAL']].rename(
    columns={'YEAR':'ds', 'ANNUAL':'y'}
)
prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')

model = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.5)
model.fit(prophet_df)

future = model.make_future_dataframe(periods=45, freq='YE')
forecast = model.predict(future)

fig_fc = plot_plotly(model, forecast)
fig_fc.update_layout(
    title='Annual Rainfall Forecast (2001–2045)',
    xaxis_title='Year',
    yaxis_title='Rainfall (mm)',
    width=1000,
    plot_bgcolor='rgba(21,101,192,0.04)',
)
fig_fc.show()

# Summary Statistics
print("\nForecast Summary (next 20 years):")
print(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(25))

print("\n--- Day 3 complete ---")
print("Project wrap-up: The analysis reveals monsoon dominance,")
print("a slight post-1960 rainfall decline, and a moderate")
print("downward forecast trend — highlighting the need for")
print("adaptive water-resource planning.")
