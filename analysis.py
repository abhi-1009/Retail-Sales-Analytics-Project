# Importing libraries
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Downloading dataset
tran = pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/RETAILCHAINANALYSIS/Retail_Data_Transactions.csv")
tran.head()
response= pd.read_csv("C:/Users/Hp/OneDrive/Desktop/python/RETAILCHAINANALYSIS/Retail_Data_Response.csv")
response.head()

# merging both dataset into one dataframe
df = tran.merge(response, on ='customer_id', how ='left')

#features
df.dtypes
df.shape
df.describe

# MISSING VALUES
df.isnull().sum()
df=df.dropna()

# change dtypes
df['trans_date']= pd.to_datetime(df['trans_date'])
df['response']= df['response'].astype('int64')

# checking outliers by using Z-Score

# calc Z Score for 'tran_amount'
z_scores= np.abs(stats.zscore(df['tran_amount']))
# set a threshold
threshold= 3
outliers= z_scores>threshold
df[outliers]
sns.boxplot(x=df['tran_amount'])
plt.title("Outlier Detection - Transaction Amount")
plt.savefig("Outlier Detection - Transaction Amount.png")
plt.show()

# calc Z Score for 'response'
z_scores= np.abs(stats.zscore(df['response']))
# set a threshold
threshold= 3
outliers= z_scores>threshold
df[outliers]
sns.boxplot(x=df['response'])
plt.title("Outlier Detection - Response")
plt.savefig("Outlier Detection - Response.png")
plt.show()

# Extract Month & Year

# creating new columns

df['total_sales_value'] = df['tran_amount']
df['year'] = df['trans_date'].dt.year
df['month']= df['trans_date'].dt.month
df['quarter'] = df['trans_date'].dt.quarter
df['day_of_week'] = df['trans_date'].dt.day_name()

# Analysis

# Yearly sales trend
yearly_sales = df.groupby('year')['tran_amount'].sum().reset_index()
print(yearly_sales)

# Best Performing Quarter
quarter_sales = df.groupby('quarter')['tran_amount'].sum().reset_index()
print(quarter_sales)

# Best performing month
monthly_sales = df.groupby(["year","month"])["tran_amount"].sum().reset_index()
monthly_sales = monthly_sales.sort_values(by='tran_amount', ascending=False)
print(monthly_sales)

# Highest Sales Day of Week
weekday_sales = df.groupby('day_of_week')['tran_amount'].sum().reset_index()
print(weekday_sales)

# Average Transaction Value
avg_transaction = df['tran_amount'].mean()
print(avg_transaction)

# Response Rate Analysis
response_rate = df['response'].value_counts(normalize=True) * 100
print(response_rate)

# Top 5 Months by Average Sales
monthly_avg_sales = df.groupby('month')['tran_amount'].mean().reset_index()
print(monthly_avg_sales)

# Customer Lifetime Value (CLV)
customer_ltv = df.groupby('customer_id')['tran_amount'].sum().reset_index()
print(customer_ltv)

# Customers With Repeat Purchases
repeat_customers = df['customer_id'].value_counts()
repeat_customers = repeat_customers[repeat_customers > 1]
print(repeat_customers)

# Monthly Response Trend
monthly_response = df.groupby('month')['response'].mean().reset_index()
print(monthly_response)

# customers having highest number of orders
customer_counts= df['customer_id'].value_counts().reset_index()
customer_counts.columns=['customer_id','count']
print(customer_counts)
# sort
top_5_cust= customer_counts.sort_values(by='count', ascending=False).head(5)
print(top_5_cust)
sns.barplot(x='customer_id', y='count', data=top_5_cust)
plt.show()

# customers having highest value of orders
customer_sales= df.groupby('customer_id')['tran_amount'].sum().reset_index()
print(customer_sales)
# sort
top_5_sales= customer_sales.sort_values(by='tran_amount', ascending=False).head(5)
print(top_5_sales)
sns.barplot(x='customer_id', y='tran_amount', data=top_5_sales)
plt.show()

# Visualization

# Monthly Sales Trend
monthly_sales = df.groupby('month')['tran_amount'].sum().reset_index()
plt.figure(figsize=(10,6))
sns.lineplot(x='month', y='tran_amount', data=monthly_sales, marker='o')
plt.title("Monthly Sales Trend")
plt.savefig("Monthly_Sales_Trend.png")
plt.show()

# Customer Response Distribution
plt.figure(figsize=(6,5))
sns.countplot(x='response', data=df)
plt.title("Customer Response Distribution")
plt.savefig("Customer Response Distribution.png")
plt.show()

# Top 10 Customers by Sales
top_customers = df.groupby('customer_id')['tran_amount'].sum().reset_index()
top_customers = top_customers.sort_values(by='tran_amount', ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(x='customer_id', y='tran_amount', data=top_customers)
plt.xticks(rotation=45)
plt.title("Top 10 Customers by Sales")
plt.savefig("Top 10 Customers by Sales.png")
plt.show()

# Transaction Amount Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['tran_amount'], bins=30, kde=True)
plt.title("Transaction Amount Distribution")
plt.xlabel("Transaction Amount")
plt.ylabel("Frequency")
plt.show()

# Response vs Transaction Amount
plt.figure(figsize=(8,5))
sns.boxplot(x='response', y='tran_amount', data=df)
plt.title("Transaction Amount vs Customer Response")
plt.savefig("Transaction Amount vs Customer Response.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df[['tran_amount','response']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("Correlation Heatmap.png")
plt.show()

# Transaction Count Per Customer
customer_counts = df['customer_id'].value_counts().head(10)
plt.figure(figsize=(10,6))
sns.barplot(x=customer_counts.index, y=customer_counts.values)
plt.title("Top Customers by Number of Transactions")
plt.xticks(rotation=45)
plt.show()
 
#Advanced Analytics

#Time Series Analysis
df['month_year']=df['trans_date'].dt.to_period('M')
monthly_sales = df.groupby('month_year')['tran_amount'].sum()
monthly_sales.index= monthly_sales.index.to_timestamp()
plt.figure(figsize=(12,6))
plt.plot(monthly_sales.index, monthly_sales.values)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xlabel('Month-Year')
plt.ylabel('Sales')
plt.title('Monthly Sales')
plt.xticks(rotation=45)
plt.show()

# Cohort Segmentation
#Recency
today = df['trans_date'].max()
recency = (today - df.groupby('customer_id')['trans_date'].max()).dt.days
#Frequency
frequency=df.groupby('customer_id')['trans_date'].count()
#Monetary
monetary= df.groupby('customer_id')['tran_amount'].sum()
#Combine
rfm=pd.DataFrame({'recency':recency, 'frequency':frequency, 'monetary':monetary})

plt.figure(figsize=(8,6))
sns.scatterplot(x='frequency', y='monetary', data=rfm)
plt.title("Customer Frequency vs Monetary Value")
plt.savefig("Customer Frequency vs Monetary Value.png")
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='recency', y='frequency', data=rfm)
plt.title("Customer Recency vs Frequency")
plt.savefig("Customer Recency vs Frequency.png")
plt.show()

# Customer Segmentation
def segment_customer(row):
    
    if row['recency']<= 30 and row['frequency']>=15 and row['monetary']>1000:
        return'P0' # Best Customers
    elif(30 <=row['recency']<= 90) and (8 <= row['frequency']<15) and (500<=row['monetary']<=1000):
        return'P1'# Medium Customers
    else:
        return'P2'# Low / At Risk Customers
rfm['Segment']= rfm.apply(segment_customer, axis=1)
print(rfm)

# Churn Analysis
# count the numbers of churned and active customers
churn_counts=df['response']. value_counts()
churn_counts.plot(kind='bar')
plt.show()

# Analysing top customers
top_5_cust=monetary.sort_values(ascending=False).head(5).index
top_customers_df=df[df['customer_id'].isin(top_5_cust)]
top_customer_sales=top_customers_df.groupby(['customer_id','month_year'])['tran_amount'].sum().unstack(level=0)
top_customer_sales.plot(kind='line')
plt.show()

df.to_csv('MainData.csv')
rfm.to_csv('AddAnalysis.csv')