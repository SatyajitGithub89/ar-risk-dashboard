# AR Risk Monitoring Dashboard

This project is an interactive dashboard built using Python and Streamlit to analyze Accounts Receivable (AR) data and identify risk patterns at the project level.

## Problem Statement
Manual analysis of receivables data is time-consuming and often relies on static rules (e.g., aging thresholds), which may not capture complex customer behavior.

## Solution
This dashboard combines rule-based logic and machine learning (KMeans clustering) to provide a more comprehensive view of receivables risk.

## Key Features
- Rule-based risk classification using aging thresholds
- KMeans clustering to identify behavioral segments
- Comparison of rule-based vs ML-based risk
- Identification of “Review Needed” cases where both approaches disagree
- Interactive filters for dynamic analysis
- Real-time KPIs and visualizations

## Business Impact
- Reduced manual analysis time from hours to minutes
- Improved identification of high-risk and anomalous receivables
- Enabled targeted review of cases where traditional rules may fail

## Tech Stack
- Python (Pandas, Scikit-learn)
- Streamlit
- Plotly

## Deployment
The application is deployed on Streamlit Cloud and can be accessed via a web interface for real-time analysis.

## Key Insight
Clustering revealed distinct behavioral segments such as high exposure vs delayed payment groups, highlighting that risk is multi-dimensional and not solely dependent on aging.
