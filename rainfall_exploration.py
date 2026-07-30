import os
import sys
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Data Loading (works in Colab & locally, auto-downloads if missing)
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

print("Shape:", rainfall_data.shape)
print(rainfall_data.head())
print(rainfall_data.info())


# Annual Rainfall Trend
annual_mean = rainfall_data['ANNUAL'].mean()

fig_annual = go.Figure()
fig_annual.add_trace(go.Scatter(
    x=rainfall_data['YEAR'],
    y=rainfall_data['ANNUAL'],
    mode='lines',
    name='Annual Rainfall',
    line=dict(color='blue', width=2)
))
fig_annual.add_hline(
    y=annual_mean,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Mean: {annual_mean:.1f} mm',
    annotation_position='top right'
)
fig_annual.update_layout(
    title='Trend in Annual Rainfall in India Since 1901',
    xaxis_title='Year', yaxis_title='Rainfall (mm)',
    height=400, width=700,
    plot_bgcolor='rgba(21,101,192,0.04)',
)
fig_annual.show()


# Average Monthly Rainfall
monthly_cols = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
monthly_avg = rainfall_data[monthly_cols].mean()
monthly_mean = monthly_avg.mean()

highest_month = monthly_avg.idxmax()
lowest_month  = monthly_avg.idxmin()

fig_monthly = px.bar(
    x=monthly_avg.index,
    y=monthly_avg.values,
    labels={'x':'Month', 'y':'Average Rainfall (mm)'},
    title='Average Monthly Rainfall in India 1901-2015',
    text=monthly_avg.values
)
fig_monthly.add_hline(
    y=monthly_mean,
    line_dash='dash',
    line_color='red',
    annotation_text=f'Mean: {monthly_mean:.1f} mm',
    annotation_position='top right'
)
fig_monthly.update_traces(marker_color='sky blue', marker_line_width=1)
fig_monthly.update_layout(plot_bgcolor='rgba(21,101,192,0.04)')
fig_monthly.show()

print(f"Highest avg rainfall month: {highest_month} ({monthly_avg.max():.1f} mm)")
print(f"Lowest  avg rainfall month: {lowest_month} ({monthly_avg.min():.1f} mm)")


# Seasonal Distribution
seasonal_cols = ['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
seasonal_avg = rainfall_data[seasonal_cols].mean()

fig_seasonal = px.bar(
    x=seasonal_avg.index,
    y=seasonal_avg.values,
    labels={'x':'Season', 'y':'Rainfall (mm)'},
    title='Seasonal Rainfall Distribution in India 1901-2015',
    text=seasonal_avg.values,
    color=seasonal_avg.values,
    color_continuous_scale=['skyblue','blue','darkblue']
)
fig_seasonal.update_traces(marker_line_width=2)
fig_seasonal.update_layout(coloraxis_colorbar=dict(title='Rainfall (mm)'), plot_bgcolor='rgba(21,101,192,0.04)')
fig_seasonal.show()

