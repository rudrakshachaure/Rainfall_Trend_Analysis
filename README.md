# Rainfall Trend Analysis — India (1901–2015)

Analysis of over a century of rainfall data across India, exploring long-term trends, seasonal patterns, extreme events, and future forecasts — wrapped in an interactive Streamlit dashboard.

## Overview

This project examines monthly, seasonal, and annual rainfall measurements spanning **1901–2015** to uncover variability, detect anomalies, and predict future rainfall trends using time-series modeling.

**Data source:** Indian Meteorological Department  
**Format:** 115 rows · 19 columns (monthly, seasonal, annual rainfall)

## Project Structure

```
├── streamlit_app.py              # Interactive dashboard (main entry point)
├── rainfall_exploration.py       # Data loading, annual trend, monthly/seasonal charts
├── climate_impact_analysis.py    # Rolling avg, drought/extreme years, correlations
├── forecasting_conclusion.py     # K-Means clustering, Prophet forecast
├── .streamlit/config.toml        # Streamlit theme config
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
├── .gitignore
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app auto-downloads the dataset from GitHub if the CSV is not present locally.

## Dashboard Pages

| Page | Description |
|------|-------------|
| 📊 **Dashboard** | 2026 summary |
| 📈 **Trends** | Annual trend, monthly averages, seasonal distribution |
| 🌦 **Climate Insights** | 10-year rolling avg, extreme years, seasonal correlations |
| 🔮 **Forecasting** | K-Means clustering + Prophet forecast with evaluation metrics |

## Key Findings

| Finding | Detail |
|---------|--------|
| **Monsoon dominance** | June–September contributes ~75% of annual rainfall |
| **Post-1960 decline** | 10-year rolling average shows a slight downward trend |
| **Extreme years** | 5 drought years and 7 extreme rainfall years detected |
| **Monsoon correlation** | 0.93 correlation with annual totals |
| **Forecast** | Prophet predicts relatively stable rainfall through ~2045 |
| **Model accuracy** | MAE: ~130mm, RMSE: ~160mm, MAPE: ~9% (test: 2001–2015) |

## Requirements

- Python 3.9+
- pandas, numpy, plotly, streamlit
- scikit-learn, scipy, prophet

## License

MIT
