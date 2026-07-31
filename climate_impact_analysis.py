# -*- coding: utf-8 -*-
"""
Day 2 — Climate Impact & Extreme Event Analysis
=================================================
Builds on Day 1 data. Covers:
  - 10-year rolling average to smooth long-term trends
  - Drought & extreme-rainfall year detection (1.5 σ thresholds)
  - Correlation between seasonal and annual rainfall
  - Correlation between monsoon and other seasons
"""

import os
import sys
import io
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr

# ---------------------------------------------------------------------------
# 1. Data Loading (auto-downloads if missing)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# 2. 10-Year Rolling Average
# ---------------------------------------------------------------------------
rainfall_data['10-Year Rolling Avg'] = rainfall_data['ANNUAL'].rolling(window=10).mean()

fig_climate = go.Figure()
fig_climate.add_trace(go.Scatter(
    x=rainfall_data['YEAR'], y=rainfall_data['ANNUAL'],
    mode='lines', name='Annual Rainfall',
    line=dict(color='blue', width=2)
))
fig_climate.add_trace(go.Scatter(
    x=rainfall_data['YEAR'], y=rainfall_data['10-Year Rolling Avg'],
    mode='lines', name='10-Year Rolling Average',
    line=dict(color='red', width=2)
))
fig_climate.update_layout(
    title='Impact of Climate Change on Rainfall in India 1901-2015',
    xaxis_title='Year', yaxis_title='Rainfall (mm)',
    height=500, width=1000,
    plot_bgcolor='rgba(21,101,192,0.04)',
)
fig_climate.show()

# ---------------------------------------------------------------------------
# 3. Drought & Extreme Rainfall Years
# ---------------------------------------------------------------------------
mean_rainfall = rainfall_data['ANNUAL'].mean()
std_dev       = rainfall_data['ANNUAL'].std()
threshold     = 1.5 * std_dev

drought_years  = rainfall_data[rainfall_data['ANNUAL'] < (mean_rainfall - threshold)]
extreme_years  = rainfall_data[rainfall_data['ANNUAL'] > (mean_rainfall + threshold)]

print("Drought Years:")
print(drought_years[['YEAR','ANNUAL']].reset_index(drop=True))
print("\nExtreme Rainfall Years:")
print(extreme_years[['YEAR','ANNUAL']].reset_index(drop=True))

# ---------------------------------------------------------------------------
# 4. Seasonal vs Annual Correlation
# ---------------------------------------------------------------------------
seasonal_cols = ['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']
seasonal_corr = {
    s: pearsonr(rainfall_data[s], rainfall_data['ANNUAL'])[0]
    for s in seasonal_cols
}
corr_df = pd.DataFrame.from_dict(seasonal_corr, orient='index', columns=['Correlation'])
print("\nSeasonal Correlation with Annual Total:")
print(corr_df)

# ---------------------------------------------------------------------------
# 5. Monsoon vs Other Seasons Correlation
# ---------------------------------------------------------------------------
monsoon = 'Jun-Sep'
non_monsoon = [c for c in seasonal_cols if c != monsoon]
relationships = {
    s: pearsonr(rainfall_data[monsoon], rainfall_data[s])[0]
    for s in non_monsoon
}

rel_df = pd.DataFrame({
    'Season': list(relationships.keys()),
    'Correlation Coefficient': list(relationships.values())
})

fig_corr = px.bar(
    rel_df, x='Season', y='Correlation Coefficient',
    title='Correlation Between Monsoon (Jun-Sep) and Other Seasons',
    text='Correlation Coefficient',
    color='Correlation Coefficient',
    color_continuous_scale='Blues'
)
fig_corr.add_hline(
    y=0, line_dash='dash', line_color='red',
    annotation_text='No Correlation', annotation_position='bottom left'
)
fig_corr.update_traces(marker_line_color='black', marker_line_width=1)
fig_corr.update_layout(plot_bgcolor='rgba(21,101,192,0.04)')
fig_corr.show()

print("\n--- Day 2 complete ---")
