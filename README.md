#  Marketing 47: Diagnostic & Optimization Engine

This project is a complete Data Science tool for Marketing Analytics, designed to help businesses forecast sales, optimize ad budget allocation, and automatically detect anomalies.

---

##  Business Problem

| Problem | Solution |
|---------|----------|
| Unknown channel effectiveness | Coefficient analysis for each channel |
| Suboptimal budget allocation | Find optimal spend ratio between channels |
| Late detection of issues | Automatic alerts when sales deviate from forecast |
| Inaccurate sales forecasting | Forecasting model with MAPE < 10% |

---

##  Project Architecture (5 Modules)

### Module 1: Data Processor
- Load and clean marketing data
- Handle missing values
- Feature engineering (lag, rolling, interaction features)
- Calculate marketing metrics (CPC, ROAS, ROI, CR)

### Module 2: Core Model
- Time series train/test split (no data leakage)
- Compare multiple models: Ridge, Huber, ElasticNet, Lasso
- TimeSeries Cross-Validation
- Feature importance analysis

### Module 3: Optimization Engine
- Grid search for optimal budget allocation
- Scenario comparison (50/50, 70/30, 30/70, etc.)
- Promo impact analysis
- Different budget level analysis

### Module 4: Diagnostic Logic
- Actual vs Forecast comparison
- Automated alert generation based on error thresholds
- Actionable insights and recommendations
- Visualization dashboard

### Module 5: Streamlit Dashboard
- Interactive web interface
- Real-time forecast visualization
- Alert monitoring
- Budget optimization tool

---

##  Model Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| R² Score | 0.95 | Excellent fit |
| MAPE | 1.34% | Very accurate |
| MAE | ~25 | Low absolute error |

*Note: Performance may vary with real-world data. This is based on sample dataset.*

---

##  Installation

### Prerequisites
- Python 3.8+
- pip or conda

## How to Use
For Data Scientists
Run 01_Data_Processor.ipynb to prepare data

Run 02_Core_Model.ipynb to train and compare models

Run 03_Optimization_Engine.ipynb to find optimal budget

Run 04_Diagnostic_Logic.ipynb to generate alerts

## For Business Users
Launch Streamlit app: streamlit run app.py

Upload your marketing data (CSV format)

View forecasts, alerts, and optimization results

## Key Learnings & Best Practices

No Data Leakage: Always split train/test chronologically

Scale Features: Use StandardScaler for numeric features only

Promo is Binary: Don't scale categorical variables

TimeSeries CV: Use TimeSeriesSplit, not random KFold

Seasonal Forecast: Simple seasonal naive often beats complex models
