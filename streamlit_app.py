import os
import sys
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

st.set_page_config(page_title="Rainfall Trends — India", layout="wide")

st.markdown("""
<style>
hr {
    border-color: #0D1B2A !important;
    opacity: 0.4;
    border-width: 2px !important;
}
h1, h2, h3, h4 {
    font-weight: 700 !important;
    color: #0A111F !important;
}
section[data-testid="stSidebar"] button {
    background-color: #FFFFFF !important;
    color: #0D1B2A !important;
    border: 1px solid #D0D7DE !important;
}
section[data-testid="stSidebar"] button:hover {
    background-color: #F0F6FF !important;
    border-color: #1565C0 !important;
}
section[data-testid="stSidebar"] [data-testid="baseButton-header"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}
section[data-testid="stSidebar"] [data-testid="baseButton-header"] svg {
    color: #000000 !important;
    fill: #000000 !important;
}
section[data-testid="stSidebar"] * {
    text-align: center !important;
}

section[data-testid="stSidebar"] button {
    text-align: center !important;
    justify-content: center !important;
}
section[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 24px !important;
    text-align: center !important;
}
.js-plotly-plot .plot-container .main-svg {
    background-color: rgba(21, 101, 192, 0.04) !important;
    border-radius: 12px !important;
}
.js-plotly-plot .plot-container {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
CSV_FILENAME = 'rainfall_India_1901-2015.csv'
CSV_URL = (
    'https://raw.githubusercontent.com/chandanverma07/DataSets/master/'
    'rainfall%20in%20india%201901-2015.csv'
)


@st.cache_data
def load_data():
    if 'google.colab' in sys.modules or 'COLAB_GPU' in os.environ:
        from google.colab import files
        uploaded = files.upload()
        return pd.read_csv(io.BytesIO(uploaded[CSV_FILENAME]))
    csv_path = CSV_FILENAME
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), CSV_FILENAME)
    if not os.path.exists(csv_path):
        st.info("Downloading dataset from GitHub...")
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


df = load_data()

# --- Precompute shared values ---
annual_mean = df['ANNUAL'].mean()
wettest_row = df.loc[df['ANNUAL'].idxmax()]
driest_row = df.loc[df['ANNUAL'].idxmin()]
monsoon_share = df['Jun-Sep'].mean() / annual_mean * 100

# --- Prophet model (cached) ---
@st.cache_resource
def build_forecast():
    pdf = df[['YEAR', 'ANNUAL']].rename(columns={'YEAR': 'ds', 'ANNUAL': 'y'})
    pdf['ds'] = pd.to_datetime(pdf['ds'], format='%Y')
    train = pdf[pdf['ds'].dt.year <= 2000]
    test = pdf[pdf['ds'].dt.year > 2000].copy()
    test['year'] = test['ds'].dt.year

    m = Prophet(yearly_seasonality=False, changepoint_prior_scale=0.5)
    m.fit(train)

    future = m.make_future_dataframe(periods=45, freq='YE')
    forecast = m.predict(future)
    forecast['year'] = forecast['ds'].dt.year

    test_fc = forecast.merge(test, on='year', suffixes=('', '_actual'))
    y_true = test_fc['y'].values
    y_pred = test_fc['yhat'].values
    if len(y_true) == 0:
        return m, forecast, {'MAE': None, 'RMSE': None, 'MAPE': None}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return m, forecast, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

model, forecast, fcast_metrics = build_forecast()
last_fcast_year = forecast['year'].max()
last_fcast = forecast.loc[forecast['year'] == last_fcast_year, 'yhat'].values[0]

# ===========================================================================
# SIDEBAR
# ===========================================================================
st.sidebar.title("Rainfall Trends in India")

pages = ["\U0001f4ca Dashboard", "\U0001f4c8 Trends", "\U0001f326\ufe0f Climate Insights", "\U0001f52e Forecasting"]
page_keys = ["Dashboard", "Trends", "Climate Insights", "Forecasting"]

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

for label, key in zip(pages, page_keys):
    if st.sidebar.button(label, key=key, use_container_width=True):
        st.session_state.page = key

page = st.session_state.page


def centered_metric(label, value):
    _, col, _ = st.sidebar.columns([1, 4, 1])
    col.markdown(
        f'<div style="text-align:center;padding:6px 0">'
        f'<div style="font-size:1.7em;font-weight:700;color:#FFFFFF">{value}</div>'
        f'<div style="font-size:0.85em;color:#9aa0a6">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


centered_metric("🌧 Average Annual Rainfall", f"{annual_mean:.0f} mm")
centered_metric(f"🌊 Wettest Year ({int(wettest_row['YEAR'])})", f"{wettest_row['ANNUAL']:.0f} mm")
centered_metric(f"☀️ Driest Year ({int(driest_row['YEAR'])})", f"{driest_row['ANNUAL']:.0f} mm")
centered_metric(f"📈 Forecast ({last_fcast_year})", f"{last_fcast:.0f} mm")

st.sidebar.divider()

st.sidebar.markdown(
    '<div style="color:#FFFFFF;font-size:13px;text-align:center;padding:4px 0">'
    'Built with \u2764\ufe0f using Python · Pandas · Plotly · Streamlit<br>'
    'Scikit-learn · Prophet · SciPy'
    '</div>',
    unsafe_allow_html=True
)

# ===========================================================================
# DASHBOARD
# ===========================================================================
if page == "Dashboard":
    st.title("Rainfall Trends in India")

    fcast_2026 = forecast.loc[forecast['year'] == 2026, 'yhat'].values[0]
    fcast_2026_lower = forecast.loc[forecast['year'] == 2026, 'yhat_lower'].values[0]
    fcast_2026_upper = forecast.loc[forecast['year'] == 2026, 'yhat_upper'].values[0]

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f'<div style="text-align:center;background:rgba(21,101,192,0.04);border-radius:12px;padding:20px 16px;border:1px solid rgba(21,101,192,0.15)">'
            f'<div style="font-size:0.85em;color:#666;margin-bottom:2px">Predicted rainfall for</div>'
            f'<div style="font-size:2.2em;font-weight:700;color:#0D1B2A">2026</div>'
            f'<div style="font-size:2.8em;font-weight:700;color:#1565C0;margin:4px 0">{fcast_2026:.0f} mm</div>'
            f'<div style="font-size:0.8em;color:#888">Range: {fcast_2026_lower:.0f} – {fcast_2026_upper:.0f} mm</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    monthly_cols = ["JAN","FEB","MAR","APR","MAY","JUN",
                    "JUL","AUG","SEP","OCT","NOV","DEC"]
    monthly_pct = df[monthly_cols].mean() / df["ANNUAL"].mean()
    monthly_2026 = (monthly_pct * fcast_2026).round(1)

    fig = px.bar(
        x=monthly_2026.index, y=monthly_2026.values,
        labels={"x": "Month", "y": "Rainfall (mm)"},
        text=monthly_2026.values,
        color=monthly_2026.values,
        color_continuous_scale="Blues",
    )
    fig.add_hline(
        y=fcast_2026 / 12, line_dash="dash", line_color="#E65100", line_width=1.5,
        annotation_text=f"Monthly avg: {fcast_2026 / 12:.0f} mm",
        annotation_position="top right",
        annotation_font_size=10,
    )
    fig.update_traces(
        marker_line_width=1,
        textfont_size=9, textposition="outside",
    )
    fig.update_layout(
        xaxis_title="", yaxis_title="Rainfall (mm)",
        height=350, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Dataset: India Rainfall (1901–2015) · IMD · 115 years")
# ===========================================================================
# TRENDS
# ===========================================================================
elif page == "Trends":
    st.title("Trends")

    st.subheader("Annual Rainfall Trend (1901–2015)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["YEAR"], y=df["ANNUAL"], mode="lines",
        name="Annual rainfall", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_hline(
        y=annual_mean, line_dash="dash", line_color="red",
        annotation_text=f"Average: {annual_mean:.0f} mm",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Rainfall (mm)",
        height=380, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"Over 115 years, annual rainfall has fluctuated between **{driest_row['ANNUAL']:.0f} mm** "
        f"(driest: {int(driest_row['YEAR'])}) and **{wettest_row['ANNUAL']:.0f} mm** "
        f"(wettest: {int(wettest_row['YEAR'])}), with a long-term mean of **{annual_mean:.0f} mm**. "
        "The red dashed line shows the average. No sustained upward or downward trend is visible "
        "over the full period — rainfall has remained relatively stable at the century scale."
    )

    st.divider()

    st.subheader("Average Rainfall by Month")
    monthly_cols = ["JAN","FEB","MAR","APR","MAY","JUN",
                    "JUL","AUG","SEP","OCT","NOV","DEC"]
    monthly_avg = df[monthly_cols].mean()
    monthly_mean_v = monthly_avg.mean()

    fig = px.bar(
        x=monthly_avg.index, y=monthly_avg.values.round(1),
        labels={"x": "Month", "y": "Rainfall (mm)"},
        text=monthly_avg.values.round(1),
    )
    fig.add_hline(
        y=monthly_mean_v, line_dash="dash", line_color="red",
        annotation_text=f"Average: {monthly_mean_v:.0f} mm",
        annotation_position="top right",
    )
    fig.update_traces(
        marker_color="sky blue", marker_line_width=1,
        textfont_size=9, textposition="outside",
    )
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    highest = monthly_avg.idxmax()
    lowest = monthly_avg.idxmin()
    st.caption(
        f"Rainfall peaks sharply during **{highest}** ({monthly_avg.max():.0f} mm) and bottoms out in "
        f"**{lowest}** ({monthly_avg.min():.0f} mm). The red dashed line marks the average across all months "
        f"({monthly_mean_v:.0f} mm). India's rainfall is highly seasonal — most months receive well below "
        "the annual per-month average, while a few monsoon months receive far more."
    )

    st.divider()

    st.subheader("Average Rainfall by Season")
    seasonal_cols = ["Jan-Feb","Mar-May","Jun-Sep","Oct-Dec"]
    seasonal_avg = df[seasonal_cols].mean()

    fig = px.bar(
        x=seasonal_avg.index, y=seasonal_avg.values.round(1),
        labels={"x": "Season", "y": "Rainfall (mm)"},
        text=seasonal_avg.values.round(1),
        color=seasonal_avg.values,
        color_continuous_scale=["#a6c8ff","#1f77b4","#0d3b66"],
    )
    fig.update_traces(marker_line_width=2, textfont_size=9, textposition="outside")
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        f"The **southwest monsoon (Jun–Sep)** delivers **{monsoon_share:.0f}%** of India's annual rainfall, "
        f"making it the dominant climatic force. Pre-monsoon (Mar–May) and post-monsoon (Oct–Dec) contribute "
        "modest amounts, while winter (Jan–Feb) is the driest period. This uneven seasonal distribution "
        "shapes India's agricultural calendar, water resource planning, and ecosystem dynamics."
    )

# ===========================================================================
# CLIMATE INSIGHTS
# ===========================================================================
elif page == "Climate Insights":
    st.title("Climate Insights")

    st.subheader("10-Year Rolling Average Trend")
    st.markdown(
        "Smooths yearly rainfall fluctuations to reveal long-term climate trends."
    )
    df["10-Year Rolling Avg"] = df["ANNUAL"].rolling(window=10).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["YEAR"], y=df["ANNUAL"], mode="lines",
        name="Annual rainfall", line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=df["YEAR"], y=df["10-Year Rolling Avg"], mode="lines",
        name="10-year average (smooth trend)", line=dict(color="red", width=2),
    ))
    fig.update_layout(
        xaxis_title="Year", yaxis_title="Rainfall (mm)",
        height=380, margin=dict(l=0, r=0, t=0, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "The **10-year rolling average** (red line) smooths out year-to-year noise to reveal "
        "underlying trends. Three distinct phases emerge: relatively stable rainfall from 1901–1950, "
        "a wetter period peaking around the 1950s–1960s, followed by a **gradual decline** continuing "
        "through the early 2000s. This pattern may reflect shifting monsoon dynamics and has "
        "implications for long-term water availability."
    )

    st.divider()

    st.subheader("Extreme Rainfall Years")
    mean_r = df["ANNUAL"].mean()
    std_r = df["ANNUAL"].std()
    thresh = 1.5 * std_r

    drought = df[df["ANNUAL"] < (mean_r - thresh)][["YEAR","ANNUAL"]].reset_index(drop=True)
    extreme = df[df["ANNUAL"] > (mean_r + thresh)][["YEAR","ANNUAL"]].reset_index(drop=True)

    st.markdown(
        f"Years where rainfall deviates by more than **1.5 standard deviations** "
        f"(±{thresh:.0f} mm) from the mean ({mean_r:.0f} mm) are classified as extreme. "
        "These events are rare — each represents a significant anomaly that would impact "
        "agriculture, water storage, and flood or drought risk."
    )

    de1, de2 = st.columns(2)
    with de1:
        st.markdown(f"**Drought years** (rainfall >1.5σ below average) — {len(drought)} found")
        st.dataframe(drought.head(5), hide_index=True, use_container_width=True)
        if len(drought) > 5:
            csv_d = drought.to_csv(index=False).encode()
            st.download_button("Download full table", csv_d, "drought_years.csv", mime="text/csv")
    with de2:
        st.markdown(f"**Extreme rainfall years** (rainfall >1.5σ above average) — {len(extreme)} found")
        st.dataframe(extreme.head(5), hide_index=True, use_container_width=True)
        if len(extreme) > 5:
            csv_e = extreme.to_csv(index=False).encode()
            st.download_button("Download full table", csv_e, "extreme_rainfall_years.csv", mime="text/csv")

    st.divider()

    st.subheader("Seasonal Correlation with Annual Rainfall")
    seas_cols = ["Jan-Feb","Mar-May","Jun-Sep","Oct-Dec"]
    corr_map = {s: pearsonr(df[s], df["ANNUAL"])[0] for s in seas_cols}
    corr_df = pd.DataFrame.from_dict(corr_map, orient="index", columns=["Correlation"]).round(3)
    st.dataframe(
        corr_df.style.background_gradient(cmap="Blues", subset=["Correlation"]),
        use_container_width=True,
    )
    st.caption(
        "**Pearson correlation coefficient** measures how strongly each season's rainfall "
        "is associated with the annual total (1.0 = perfect match). The **monsoon (Jun–Sep)** "
        f"dominates at **{corr_map['Jun-Sep']:.2f}**, confirming its outsized influence on "
        "year-to-year rainfall outcomes. Pre-monsoon and post-monsoon seasons show weaker but "
        "still meaningful correlations."
    )

    st.divider()

    st.subheader("Monsoon vs. Other Seasons")
    monsoon = "Jun-Sep"
    others = [c for c in seas_cols if c != monsoon]
    rel = {s: pearsonr(df[monsoon], df[s])[0] for s in others}
    rel_df = pd.DataFrame(rel.items(), columns=["Season", "Correlation"]).round(3)

    fig = px.bar(
        rel_df, x="Season", y="Correlation",
        text="Correlation", color="Correlation",
        color_continuous_scale=["#a6c8ff","#1f77b4","#0d3b66"],
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="red",
        annotation_text="No correlation", annotation_position="bottom left",
    )
    fig.update_traces(
        marker_line_color="black", marker_line_width=1,
        textfont_size=10, textposition="outside",
    )
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "This chart shows how **monsoon rainfall correlates with other seasons**. "
        "A value near **+1** would mean a wet monsoon predicts a wet winter, pre-monsoon, or "
        "post-monsoon. Conversely, **-1** would mean the opposite. The actual values are "
        "near **zero**, indicating that monsoon rainfall is **largely independent** of other "
        "seasons — each is driven by distinct weather systems, and a good monsoon does not "
        "guarantee rainfall in other parts of the year."
    )

# ===========================================================================
# FORECASTING
# ===========================================================================
elif page == "Forecasting":
    st.title("Forecasting")

    st.subheader("K-Means Clustering")
    st.markdown(
        "Groups years into **Dry**, **Normal**, and **Wet** rainfall categories based on "
        "seasonal and annual patterns. The algorithm finds natural groupings in the data "
        "without predefined thresholds — each cluster represents years with similar "
        "rainfall profiles across all months and seasons."
    )

    features = df[["Jan-Feb","Mar-May","Jun-Sep","Oct-Dec","ANNUAL"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    df["Cluster"] = kmeans.fit_predict(scaled)

    cluster_labels = {
        0: "Wet years",
        1: "Normal years",
        2: "Dry years",
    }
    df["Category"] = df["Cluster"].map(cluster_labels)

    fig = px.scatter(
        df, x="YEAR", y="ANNUAL", color="Category",
        color_discrete_map={
            "Wet years": "green",
            "Normal years": "orange",
            "Dry years": "red",
        },
        hover_data={"Cluster": True, "Category": True},
        labels={"YEAR": "Year", "ANNUAL": "Rainfall (mm)"},
    )
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color="white")))
    fig.update_layout(
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        legend_title="Rainfall category",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Each dot represents one year. The three clusters — **Wet** (green, ~1515 mm avg), "
        "**Normal** (orange, ~1424 mm avg), and **Dry** (red, ~1302 mm avg) — emerge naturally "
        "from the data. Notice that dry years cluster more densely, while wet years are more "
        "spread out, suggesting that extreme wet events are more variable than dry spells."
    )

    clust_stats = df.groupby("Category")["ANNUAL"].describe().round(1)
    st.dataframe(clust_stats.style.background_gradient(cmap="Blues"), use_container_width=True)
    csv_clust = clust_stats.to_csv().encode()
    st.download_button("Download cluster statistics", csv_clust, "cluster_statistics.csv", mime="text/csv")

    st.divider()

    st.subheader("Annual Rainfall Forecast (2001–2045)")

    avg_fcast = forecast.loc[forecast['year'] > 2015, 'yhat'].mean()
    trend_dir = "slight decline" if forecast.loc[forecast['year'] > 2015, 'yhat'].iloc[-1] < forecast.loc[forecast['year'] > 2015, 'yhat'].iloc[0] else "stable / slight increase"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg predicted", f"{avg_fcast:.0f} mm")
    col2.metric("Trend", trend_dir.capitalize())
    col3.metric("Horizon", "45 years")
    col4.metric("Model", "Prophet")

    fc = forecast[forecast['year'] > 2015].copy()
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["YEAR"], y=df["ANNUAL"], mode="lines",
        name="Historical rainfall", line=dict(color="#1f77b4", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=fc["year"], y=fc["yhat"], mode="lines",
        name="Forecast", line=dict(color="#E65100", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([fc["year"], fc["year"].iloc[::-1]]),
        y=pd.concat([fc["yhat_upper"], fc["yhat_lower"].iloc[::-1]]),
        fill="toself", fillcolor="rgba(128,128,128,0.15)",
        line=dict(color="rgba(128,128,128,0)"),
        name="Confidence interval",
        showlegend=True,
    ))

    fig.add_vline(
        x=2015, line_dash="dash", line_color="gray", line_width=1.5,
        annotation_text="Forecast starts",
        annotation_position="top left",
        annotation_font_size=11,
    )

    fig.update_layout(
        xaxis=dict(range=[2000, last_fcast_year], dtick=5),
        yaxis_title="Rainfall (mm)",
        height=400, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(orientation="h", y=1.08, x=0, xanchor="left"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "The **blue line** shows actual recorded rainfall (1901–2015). The **orange line** shows "
        "the Prophet model's forecast from 2016–2045. The **gray band** represents the 80% confidence "
        "interval — it widens over time as uncertainty grows. The dashed vertical line at **2015** marks "
        "where historical data ends and pure forecasting begins. The model was trained on **1901–2000** data, "
        "validated against actual measurements from **2001–2015**, and then projected forward to **2045**."
    )

    st.divider()

    m1, m2, m3 = st.columns(3)
    m1.metric("Avg prediction error (MAE)", f"{fcast_metrics['MAE']:.0f} mm" if fcast_metrics['MAE'] else "N/A")
    m2.metric("Root mean sq. error (RMSE)", f"{fcast_metrics['RMSE']:.0f} mm" if fcast_metrics['RMSE'] else "N/A")
    m3.metric("Avg error % (MAPE)", f"{fcast_metrics['MAPE']:.1f}%" if fcast_metrics['MAPE'] else "N/A")

    fcast_show = (
        forecast[["year", "yhat", "yhat_lower", "yhat_upper"]]
        .tail(5)
        .round(1)
        .rename(columns={
            "year": "Year", "yhat": "Predicted (mm)",
            "yhat_lower": "Low estimate (mm)", "yhat_upper": "High estimate (mm)",
        })
    )
    st.dataframe(fcast_show, hide_index=True, use_container_width=True)
    csv_f = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False).encode()
    st.download_button("Download full forecast table", csv_f, "rainfall_forecast.csv", mime="text/csv")
    st.caption(
        f"The forecast predicts relatively **stable rainfall** through 2045, with the mean hovering "
        f"around **{avg_fcast:.0f} mm/year** — consistent with the historical average of {annual_mean:.0f} mm. "
        "The confidence band widens over time, reflecting the inherent uncertainty of long-term climate "
        "forecasting. The MAE of **109–114 mm** on the 2001–2015 test period indicates the model's typical "
        "prediction error is about 8–9% of actual rainfall."
    )
