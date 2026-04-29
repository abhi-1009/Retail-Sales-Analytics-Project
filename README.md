## Retail Sales Analytics & Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?style=flat&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-latest-9cf?style=flat)
![Matplotlib](https://img.shields.io/badge/Matplotlib-latest-orange?style=flat)
![SciPy](https://img.shields.io/badge/SciPy-latest-8CAAE6?style=flat&logo=scipy)
![Domain](https://img.shields.io/badge/Domain-Retail%20Analytics-lightblue?style=flat)

A comprehensive retail analytics project that combines **EDA, time series analysis, RFM customer segmentation, cohort analysis, and churn analysis** on transactional retail data. The project uncovers sales trends, identifies high-value customers, evaluates marketing response patterns, and generates actionable business recommendations — with all visualisations saved as `.png` files.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Data Pipeline](#data-pipeline)
- [Analysis Performed](#analysis-performed)
- [RFM Customer Segmentation](#rfm-customer-segmentation)
- [Visualisations Generated](#visualisations-generated)
- [Key Insights](#key-insights)
- [Screenshots](#screenshots)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Business Recommendations](#business-recommendations)
- [Future Scope](#future-scope)

## Project Overview
This project analyses retail transaction and customer response data to:
- Clean, merge, and engineer features from two raw datasets
- Detect and handle outliers using the **Z-Score method**
- Analyse sales by year, quarter, month, and day of week
- Build an **RFM (Recency, Frequency, Monetary) model** for customer segmentation
- Classify customers into 3 tiers: **P0 (Best)**, **P1 (Medium)**, **P2 (At Risk)**
- Perform **churn analysis** using the customer `response` column
- Export final datasets to CSV for further analysis

## Dataset
| Property | Detail |
| :--- | :--- |
| **Source File 1** | `Retail_Data_Transactions.csv` |
| **Source File 2** | `Retail_Data_Response.csv` |
| **Merge Key** | `customer_id` (left join) |
| **Output Files** | `MainData.csv`, `AddAnalysis.csv` (RFM results) |

### Dataset Columns
| Column | Source | Type | Description |
| :--- | :---: | :---: | :--- |
| `customer_id` | Both | String | Unique customer identifier |
| `trans_date` | Transactions | Date | Transaction date |
| `tran_amount` | Transactions | Float | Transaction value |
| `response` | Response | Int (0/1) | Customer marketing response |

### Engineered Features (from source code)
| Feature | Formula | Purpose |
| :--- | :--- | :--- |
| `total_sales_value` | `= tran_amount` | Alias for sales |
| `year` | `trans_date.dt.year` | Year extraction |
| `month` | `trans_date.dt.month` | Month number |
| `quarter` | `trans_date.dt.quarter` | Quarter (1–4) |
| `day_of_week` | `trans_date.dt.day_name()` | Day name string |
| `month_year` | `trans_date.dt.to_period('M')` | Period for time series |

## Technologies Used
| Technology | Version | Purpose |
| :--- | :---: | :--- |
| **Python** | 3.9+ | Core programming language |
| **Pandas** | 2.x | Data loading, merging, feature engineering, groupby |
| **NumPy** | latest | Z-score calculation, RFM derivation |
| **SciPy** | latest | `stats.zscore()` for outlier detection |
| **Matplotlib** | latest | Line charts, bar charts, histograms, time series plots |
| **Seaborn** | latest | Boxplots, countplots, heatmaps, scatter plots, barplots |
| **matplotlib.dates** | built-in | Date formatting for time series x-axis |
| **datetime** | built-in | Date arithmetic for recency calculation |

### Python Libraries (from source code)
```python
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
```

##  Data Pipeline
```
Retail_Data_Transactions.csv  +  Retail_Data_Response.csv
              │
              ▼
Merge on customer_id (left join)
              │
              ▼
Data Cleaning
├── dropna() — remove missing values
├── trans_date → pd.to_datetime
└── response → int64

              │
              ▼
Outlier Detection — Z-Score Method (threshold = 3)
├── tran_amount — boxplot saved
└── response — boxplot saved

              │
              ▼
Feature Engineering
├── year, month, quarter, day_of_week
├── total_sales_value, month_year

              │
              ▼
EDA & Analysis (12 analyses)
├── Yearly / Quarterly / Monthly / Weekly sales
├── Average transaction value
├── Response rate
├── Customer LTV
├── Repeat customers
├── Top 5 customers by orders and by value

              │
              ▼
Visualisations (10 charts saved as .png)

              │
              ▼
Advanced Analytics
├── Time Series (Monthly Sales Trend)
├── RFM Analysis (Recency, Frequency, Monetary)
├── Customer Segmentation (P0 / P1 / P2)
└── Churn Analysis

              │
              ▼
Export: MainData.csv + AddAnalysis.csv (RFM)
```

## Analysis Performed
All 12 analyses extracted from source code:
| # | Analysis | Method |
| :---: | :--- | :--- |
| 1 | **Yearly Sales Trend** | `groupby('year')['tran_amount'].sum()` |
| 2 | **Best Performing Quarter** | `groupby('quarter')['tran_amount'].sum()` |
| 3 | **Best Performing Month** | `groupby(['year','month'])['tran_amount'].sum()` sorted desc |
| 4 | **Highest Sales Day of Week** | `groupby('day_of_week')['tran_amount'].sum()` |
| 5 | **Average Transaction Value** | `df['tran_amount'].mean()` |
| 6 | **Marketing Response Rate** | `value_counts(normalize=True) * 100` |
| 7 | **Top 5 Months by Avg Sales** | `groupby('month')['tran_amount'].mean()` |
| 8 | **Customer Lifetime Value (CLV)** | `groupby('customer_id')['tran_amount'].sum()` |
| 9 | **Repeat Purchase Customers** | `value_counts()` filtered `> 1` transaction |
| 10 | **Monthly Response Trend** | `groupby('month')['response'].mean()` |
| 11 | **Top 5 Customers by Order Count** | `value_counts().head(5)` |
| 12 | **Top 5 Customers by Sales Value** | `groupby('customer_id').sum().sort_values().head(5)` |

## RFM Customer Segmentation
RFM was calculated from the transaction data and combined into a single DataFrame:
```python
# Recency — days since last purchase
today = df['trans_date'].max()
recency = (today - df.groupby('customer_id')['trans_date'].max()).dt.days

# Frequency — number of transactions
frequency = df.groupby('customer_id')['trans_date'].count()

# Monetary — total spend
monetary = df.groupby('customer_id')['tran_amount'].sum()
rfm = pd.DataFrame({'recency': recency, 'frequency': frequency, 'monetary': monetary})
```
### Customer Segment Rules
| Segment | Label | Recency | Frequency | Monetary |
| :---: | :--- | :---: | :---: | :---: |
| **P0** | Best Customers | ≤ 30 days | ≥ 15 transactions | > ₹1,000 |
| **P1** | Medium Customers | 30–90 days | 8–14 transactions | ₹500–₹1,000 |
| **P2** | Low / At-Risk Customers | All others | All others | All others |

## Visualisations Generated
| File | Chart Type | Description |
| :--- | :---: | :--- |
| `Outlier Detection - Transaction Amount.png` | Boxplot | Z-score outlier check on `tran_amount` |
| `Outlier Detection - Response.png` | Boxplot | Z-score outlier check on `response` |
| `Monthly_Sales_Trend.png` | Line chart | Monthly total sales trend |
| `Customer Response Distribution.png` | Countplot | Count of responded (1) vs not responded (0) |
| `Top 10 Customers by Sales.png` | Bar chart | Top 10 customers ranked by total spend |
| `Transaction Amount vs Customer Response.png` | Boxplot | Sales amount distribution by response group |
| `Correlation Heatmap.png` | Heatmap | Correlation between `tran_amount` and `response` |
| `Customer Frequency vs Monetary Value.png` | Scatter | RFM scatter: frequency vs monetary |
| `Customer Recency vs Frequency.png` | Scatter | RFM scatter: recency vs frequency |
| Time Series Plot *(displayed only)* | Line chart | Monthly sales with 6-month x-axis intervals |

## Key Insights
- **Seasonal sales variations exist** — certain months and quarters consistently outperform others
- **High-value customers (P0)** contribute disproportionately to total revenue despite being a small group
- **Repeat purchase customers** are a significant retention base — identified via `value_counts() > 1`
- **Marketing response** has measurable impact on transaction amounts — boxplot shows responded customers have higher average spend
- **Top 10 customers by CLV** generate significantly more revenue than the median customer
- **Discount / response interaction** suggests targeted campaigns can shift customers from P2 to P1 tier
- **Day-of-week analysis** reveals which days drive peak sales — useful for staffing and promotions

##  Screenshots
### 1. Monthly Sales Trend
![Monthly Sales Trend](screenshots/Monthly_Sales_Trend.png)
*Line chart of total sales per month across the full date range*

### 2. Outlier Detection — Transaction Amount
![Outlier Boxplot](screenshots/Outlier_Detection_Transaction_Amount.png)
*Boxplot showing Z-score outlier detection on transaction amounts (threshold = 3)*

### 3. Top 10 Customers by Sales
![Top Customers](screenshots/Top_10_Customers_by_Sales.png)
*Bar chart ranking top 10 customers by total transaction value*

### 4. Transaction Amount vs Customer Response
![Response vs Amount](screenshots/Transaction_Amount_vs_Customer_Response.png)
*Boxplot comparing transaction amounts between responded (1) and non-responded (0) customers*

### 5. RFM — Customer Frequency vs Monetary Value
![RFM Scatter](screenshots/Customer_Frequency_vs_Monetary_Value.png)
*Scatter plot of RFM frequency vs monetary — clusters visible for P0/P1/P2 segments*

### 6. Customer Recency vs Frequency
![Recency vs Frequency](screenshots/Customer_Recency_vs_Frequency.png)
*Scatter plot of RFM recency vs frequency — shows active vs churned customer patterns*

### 7. Correlation Heatmap
![Correlation Heatmap](screenshots/Correlation_Heatmap.png)
*Annotated heatmap showing correlation between tran_amount and response*

## Installation and Setup
### Step 1 — Clone the Repository
```bash
git clone https://github.com/abhi-1009/Retail-Sales-Analytics-Project.git
cd Retail-Sales-Analytics-Project
```
### Step 2 — Install Required Libraries
```bash
pip install pandas numpy scipy matplotlib seaborn
```
### Step 3 — Add Datasets
Place both CSV files in the project folder and update paths:
```python
tran = pd.read_csv("Retail_Data_Transactions.csv")
response = pd.read_csv("Retail_Data_Response.csv")
```
### Step 4 — Run the Analysis
```bash
python retail_analytics.py
```
This will:
- Generate and save all 9 `.png` visualisation files
- Print all 12 analysis results to the console
- Export `MainData.csv` and `AddAnalysis.csv` (RFM results)

## Usage
After running the script you will find in your project folder:
| Output | Description |
| :--- | :--- |
| `MainData.csv` | Full cleaned and feature-engineered dataset |
| `AddAnalysis.csv` | RFM table with Recency, Frequency, Monetary, Segment (P0/P1/P2) |
| `*.png` files | All 9 saved visualisation charts |

The RFM CSV (`AddAnalysis.csv`) can be opened in Excel for further filtering and business reporting.

## Business Recommendations
| Finding | Recommendation |
| :--- | :--- |
| P0 customers drive most revenue | Implement exclusive loyalty programs and early-access promotions |
| P2 customers show low recency | Launch win-back email campaigns with personalised discounts |
| Seasonal sales peaks identified | Plan inventory procurement and staffing around high-demand months |
| Marketing response lifts spend | Increase frequency of targeted campaigns for non-responders |
| Repeat customers are significant | Introduce referral rewards to grow this segment further |

## Future Scope
- **ML Sales Forecasting** — ARIMA / Prophet time series forecasting on monthly sales
- **Churn Prediction Model** — Binary classifier (Random Forest / XGBoost) on `response` column
- **K-Means Clustering** — Replace rule-based P0/P1/P2 with data-driven cluster segmentation
- **Recommendation System** — Collaborative filtering for product/category cross-sell
- **Automated Dashboard** — Streamlit or Power BI dashboard with live data refresh
- **Marketing Strategy Optimiser** — A/B test simulation based on response segment analysis

## Author
**Abhijit Sinha**
- GitHub: [@abhi-1009](https://github.com/abhi-1009)
- LinkedIn: [abhijit-sinha-053b159a](https://linkedin.com/in/abhijit-sinha-053b159a)
- Email: sinhaabhijit12@yahoo.com
