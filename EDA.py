#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary library
import pandas as pd

# Load the CSV file (since it's in the same folder, just use the filename)
file_path = 'Stock prices and macroeconomic data.csv'
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Display the first few rows to understand the structure
print("\nFirst 5 Rows of the Dataset:")
print(data.head())


# In[3]:


# Import necessary libraries
import pandas as pd

# Load the dataset
file_path = 'Stock prices and macroeconomic data.csv'
data = pd.read_csv(file_path)

# Make a copy to avoid changing the original data
cleaned_data = data.copy()

# 1. Convert 'Date' column to datetime format
cleaned_data['Date'] = pd.to_datetime(cleaned_data['Date'])

# 2. Clean 'Vol.' column: convert strings like "9.90K" or "2.3M" into numbers
def parse_volume(val):
    if isinstance(val, str):
        val = val.replace(',', '').strip()
        if 'K' in val:
            return float(val.replace('K', '')) * 1_000
        elif 'M' in val:
            return float(val.replace('M', '')) * 1_000_000
        else:
            return float(val)
    return val

cleaned_data['Vol.'] = cleaned_data['Vol.'].apply(parse_volume)

# 3. Clean 'Change %' column: remove '%' and convert to float
cleaned_data['Change %'] = cleaned_data['Change %'].str.replace('%', '').astype(float)

# 4. Clean 'Inflation rate' column: remove '%' and convert to float
cleaned_data['Inflation rate'] = cleaned_data['Inflation rate'].str.replace('%', '')
cleaned_data['Inflation rate'] = pd.to_numeric(cleaned_data['Inflation rate'], errors='coerce')

# 5. Clean 'GDP(Rs.million)' column: remove commas and convert to numeric
cleaned_data['GDP(Rs.million)'] = cleaned_data['GDP(Rs.million)'].str.replace(',', '')
cleaned_data['GDP(Rs.million)'] = pd.to_numeric(cleaned_data['GDP(Rs.million)'], errors='coerce')

# 6. Handle missing values
# Use forward fill method to fill reasonable missing data
cleaned_data.fillna(method='ffill', inplace=True)

# 7. Sort the data by Date just to be safe
cleaned_data.sort_values(by='Date', inplace=True)
cleaned_data.reset_index(drop=True, inplace=True)

# Display cleaned dataset info
print("Cleaned Dataset Information:")
print(cleaned_data.info())

# Display first few rows of cleaned data
print("\nFirst 5 Rows of Cleaned Data:")
print(cleaned_data.head())


# In[4]:


# Basic structure
print("Dataset Info:")
print(cleaned_data.info())

# Dataset shape
print("\nDataset Shape:")
print(cleaned_data.shape)

# Check for missing values
print("\nMissing Values:")
print(cleaned_data.isnull().sum())

# Check basic statistics
print("\nSummary Statistics:")
print(cleaned_data.describe())


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Set up general plot style
sns.set(style="whitegrid")

# Plot distribution for each numerical column
numerical_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns

for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(cleaned_data[col], kde=True, color="skyblue")
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()


# In[6]:


# Plot stock prices over time
plt.figure(figsize=(12, 6))
plt.plot(cleaned_data['Date'], cleaned_data['Price'], color='blue')
plt.title('Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Plot Inflation Rate and GDP over time
fig, ax1 = plt.subplots(figsize=(12,6))

color = 'tab:red'
ax1.set_xlabel('Date')
ax1.set_ylabel('Inflation Rate (%)', color=color)
ax1.plot(cleaned_data['Date'], cleaned_data['Inflation rate'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:green'
ax2.set_ylabel('GDP (Rs. million)', color=color)
ax2.plot(cleaned_data['Date'], cleaned_data['GDP(Rs.million)'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Inflation Rate and GDP Over Time')
plt.show()


# In[7]:


pip install seaborn --upgrade


# In[8]:


# Drop 'Date' column because it's datetime and not meaningful for correlation
data_for_corr = cleaned_data.drop(columns=['Date'])

# Compute correlation matrix
corr_matrix = data_for_corr.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Numerical Features (without Date)')
plt.show()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

# 1. Assume df is your cleaned dataset (without 'Date')
numeric_cols = cleaned_data.select_dtypes(include=['float64', 'int64']).columns

# 2. Plot
plt.figure(figsize=(15, 10))

for idx, col in enumerate(numeric_cols, 1):
    plt.subplot((len(numeric_cols) + 2) // 3, 3, idx)  # 3 plots per row
    sns.boxplot(y=cleaned_data[col], color="skyblue", width=0.5)
    plt.title(f'Boxplot of {col}', fontsize=10)
    plt.tight_layout()

plt.suptitle('Boxplots of Numeric Features', fontsize=16, y=1.02)
plt.show()


# In[10]:


import pandas as pd

# Assume 'cleaned_data' is your preprocessed DataFrame (excluding 'Date' column)
outlier_counts = {}

# Loop through all numerical columns
for col in cleaned_data.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = cleaned_data[col].quantile(0.25)
    Q3 = cleaned_data[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = cleaned_data[(cleaned_data[col] < lower_bound) | (cleaned_data[col] > upper_bound)]
    
    # Store count
    outlier_counts[col] = len(outliers)

# Convert to DataFrame for better display
outlier_summary = pd.DataFrame.from_dict(outlier_counts, orient='index', columns=['Outlier Count'])
outlier_summary = outlier_summary.sort_values(by='Outlier Count', ascending=False)

print(outlier_summary)


# In[11]:


# If you want all boxplots in ONE figure horizontally
plt.figure(figsize=(14, 8))
sns.boxplot(data=cleaned_data[numeric_cols], palette="Set3")
plt.title('Boxplots of All Numeric Features', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the variables you want to plot against 'Price'
variables = ['Open', 'High', 'Low', 'Vol.', 'Change %', 'Inflation rate', 'Interest rate (%)', 'GDP(Rs.million)', 'Exchange rate']

# Create a grid of subplots
plt.figure(figsize=(18, 18))
for idx, var in enumerate(variables, 1):
    plt.subplot(3, 3, idx)
    sns.scatterplot(x=var, y='Price', data=cleaned_data)
    plt.title(f'{var} vs Stock Price', fontsize=12)
    plt.xlabel(var, fontsize=10)
    plt.ylabel('Stock Price', fontsize=10)

plt.tight_layout()
plt.suptitle('Relationships between Stock Price and Other Variables', fontsize=20, y=1.02)
plt.show()


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt

# 0. (Optional but good practice) Remove any leading/trailing spaces again
cleaned_data.columns = cleaned_data.columns.str.strip()

# 1. Drop 'Date' column if it's still there (pairplots work only with numeric data)
data_for_pairplot = cleaned_data.drop(columns=['Date'])

# 2. Create the pair plot
sns.pairplot(data_for_pairplot)
plt.suptitle('Pair Plot of Stock Prices and Macroeconomic Variables', y=1.02, fontsize=20)
plt.show()


# In[14]:


# Install statsmodels if needed
# !pip install statsmodels

from statsmodels.tsa.stattools import adfuller

# Function to check stationarity
def check_stationarity(series):
    result = adfuller(series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("✅ Data is stationary.")
    else:
        print("❌ Data is not stationary.")

# Check stationarity for 'Price'
print("Checking Stationarity of Stock Price:")
check_stationarity(cleaned_data['Price'])


# In[15]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define lag days you want to check
lags = [1, 3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]

# 2. Define macroeconomic variables
macro_vars = ['Inflation rate', 'Interest rate (%)', 'GDP(Rs.million)', 'Exchange rate']

# 3. Create lagged versions of macro variables
for var in macro_vars:
    for lag in lags:
        cleaned_data[f'{var}_lag{lag}'] = cleaned_data[var].shift(lag)

# 4. Create a DataFrame to store correlation results
corr_results = pd.DataFrame(index=lags, columns=macro_vars)

# 5. Calculate correlations between Price and lagged variables
for var in macro_vars:
    for lag in lags:
        lagged_col = f'{var}_lag{lag}'
        corr = cleaned_data[['Price', lagged_col]].corr().iloc[0, 1]
        corr_results.loc[lag, var] = corr

# 6. Convert values to float
corr_results = corr_results.astype(float)

# 7. Display correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_results.T, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Lagged Correlation of Macroeconomic Indicators with Stock Price')
plt.xlabel('Lag in Days')
plt.ylabel('Macroeconomic Indicator')
plt.tight_layout()
plt.show()


# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

# Set seasonal period (use correct one based on FFT or ACF)
seasonal_period = 20  # e.g., 5 for weekly seasonality in daily data

# Apply seasonal decomposition
decomposition = seasonal_decompose(data['Price'].dropna(), model='additive', period=seasonal_period)

# Plot the decomposition
decomposition.plot()
plt.suptitle('Seasonal Decomposition', fontsize=16)
plt.tight_layout()
plt.show()


# In[17]:


import pandas as pd
import matplotlib.pyplot as plt

# Example: Load your data
# df = pd.read_csv("your_data.csv", parse_dates=['Date'], index_col='Date')

# Let's assume your time series is in the 'Price' column
ts = data['Price']

# Set the seasonal period (e.g., 5 for weekly, 12 for monthly, 90 for quarterly depending on your dataset)
seasonal_period = 20  # Replace with the actual detected period

# Apply seasonal differencing
seasonal_diff = ts.diff(seasonal_period)

# Drop missing values introduced by differencing
seasonal_diff = seasonal_diff.dropna()

# Plot the result
plt.figure(figsize=(12, 5))
plt.plot(seasonal_diff)
plt.title(f'Seasonally Differenced Series (lag={seasonal_period})')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.grid(True)
plt.show()


# In[18]:


from statsmodels.tsa.stattools import adfuller

# Run ADF test on seasonally differenced data
result = adfuller(seasonal_diff)

# Print results
print("Augmented Dickey-Fuller Test Results:")
print(f"ADF Statistic       : {result[0]:.4f}")
print(f"p-value             : {result[1]:.4f}")
print(f"# Lags Used         : {result[2]}")
print(f"# Observations Used : {result[3]}")
print("Critical Values     :")
for key, value in result[4].items():
    print(f"   {key} : {value:.4f}")

# Interpretation
if result[1] < 0.05:
    print("\n✅ The series is likely stationary (reject H0)")
else:
    print("\n⚠️ The series is likely non-stationary (fail to reject H0)")


# In[19]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define lag days you want to check
lags = [1, 3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]

# 2. Define macroeconomic variables
var_want = ['Vol.', 'Change %']

# 3. Create lagged versions of macro variables
for var in var_want:
    for lag in lags:
        cleaned_data[f'{var}_lag{lag}'] = cleaned_data[var].shift(lag)

# 4. Create a DataFrame to store correlation results
corr_results = pd.DataFrame(index=lags, columns=var_want)

# 5. Calculate correlations between Price and lagged variables
for var in var_want:
    for lag in lags:
        lagged_col = f'{var}_lag{lag}'
        corr = cleaned_data[['Price', lagged_col]].corr().iloc[0, 1]
        corr_results.loc[lag, var] = corr

# 6. Convert values to float
corr_results = corr_results.astype(float)

# 7. Display correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_results.T, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Lagged Correlation of vol. and change % with Stock Price')
plt.xlabel('Lag in Days')
plt.ylabel('variable')
plt.tight_layout()
plt.show()


# In[20]:


import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assume `seasonal_diff` is your seasonally differenced series (from previous steps)

plt.figure(figsize=(14, 6))

# ACF Plot
plt.subplot(1, 2, 1)
plot_acf(seasonal_diff, lags=40, ax=plt.gca(), title='ACF of Seasonally Differenced Price')

# PACF Plot
plt.subplot(1, 2, 2)
plot_pacf(seasonal_diff, lags=40, ax=plt.gca(), title='PACF of Seasonally Differenced Price', method='ywm')

plt.tight_layout()
plt.show()


# In[21]:


import pandas as pd

# Assuming 'seasonal_diff' is already your seasonally differenced series
# First differencing (normal)
first_diff_after_seasonal = seasonal_diff.diff()

# Drop NaN values caused by differencing
first_diff_after_seasonal.dropna(inplace=True)

print(first_diff_after_seasonal.head())


# In[25]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Assuming 'first_diff_after_seasonal' contains the first differenced data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_acf(first_diff_after_seasonal, lags=200, ax=plt.gca())
plt.title('ACF - First Differenced Price')

plt.subplot(1, 2, 2)
plot_pacf(first_diff_after_seasonal, lags=200, ax=plt.gca(), method='ywm')
plt.title('PACF - First Differenced Price')

plt.tight_layout()
plt.show()


# In[23]:


# First differencing on seasonal differenced Price data
first_diff = seasonal_diff.diff().dropna()

# Plot the first differenced Price
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(first_diff, color='blue')
plt.title('First Differenced Price (After Seasonal Differencing)')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.grid(True)
plt.show()


# In[2]:


# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# List of numerical columns (excluding 'Date')
numerical_columns = ['Price', 'Open', 'High', 'Low', 'Volume', 'Inflation rate(%)', 'Interest rate(%)', 'GDP(Rs.million)', 'Exchange rate']

# Function to plot the correlation matrix for a given period
def plot_correlation_matrix(data, period_name):
    # Correlation matrix
    corr_matrix = data[numerical_columns].corr()
    
    # Plotting the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Correlation Matrix - {period_name}', fontsize=16)
    plt.show()
    return corr_matrix

# Pre-COVID correlation matrix
pre_covid_corr = plot_correlation_matrix(pre_covid, 'Pre-COVID')

# COVID correlation matrix
covid_corr = plot_correlation_matrix(covid, 'COVID')

# Post-COVID correlation matrix
post_covid_corr = plot_correlation_matrix(post_covid, 'Post-COVID')


# In[12]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# List of macroeconomic columns
macroeconomic_columns = ['Inflation rate(%)', 'Interest rate(%)', 'GDP(Rs.million)', 'Exchange rate']

# List of lag days
lag_days = [1, 3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220]

# Function to compute lagged correlations
def lagged_correlation(data, lag_days, macroeconomic_columns):
    correlation_results = {}

    for lag in lag_days:
        # Creating lagged features for macroeconomic indicators
        lagged_data = data.copy()
        
        # Creating lag columns for each macroeconomic indicator
        for column in macroeconomic_columns:
            lagged_data[f'{column}_lag_{lag}'] = lagged_data[column].shift(lag)
        
        # Correlation matrix with 'Price' and the lagged macroeconomic indicators
        lagged_corr = lagged_data[['Price'] + [f'{column}_lag_{lag}' for column in macroeconomic_columns]].corr()
        
        # Store correlation for each lag
        correlation_results[lag] = lagged_corr['Price'].drop('Price')
    
    # Convert dictionary to DataFrame for easier visualization
    correlation_df = pd.DataFrame(correlation_results).T
    correlation_df.index.name = 'Lag (days)'
    return correlation_df

# Function to plot lagged correlation matrix
def plot_lagged_correlation(correlation_df, period_name):
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title(f'Lagged Correlation Matrix with Price - {period_name}', fontsize=16)
    plt.xlabel('Macroeconomic Indicator', fontsize=12)
    plt.ylabel('Lag (days)', fontsize=12)
    plt.show()

# Pre-COVID lagged correlation
pre_covid_lagged_corr = lagged_correlation(pre_covid, lag_days, macroeconomic_columns)
plot_lagged_correlation(pre_covid_lagged_corr, 'Pre-COVID')

# COVID lagged correlation
covid_lagged_corr = lagged_correlation(covid, lag_days, macroeconomic_columns)
plot_lagged_correlation(covid_lagged_corr, 'COVID')

# Post-COVID lagged correlation
post_covid_lagged_corr = lagged_correlation(post_covid, lag_days, macroeconomic_columns)
plot_lagged_correlation(post_covid_lagged_corr, 'Post-COVID')


# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define lag days you want to check
lags = [1, 3, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240]

# 2. Define macroeconomic variables
macro_vars = ['Inflation rate(%)', 'Interest rate(%)', 'GDP(Rs.million)', 'Exchange rate']

# 3. Define function to calculate lagged correlations for a given period
def calculate_lagged_correlations(df, period_name):
    # Create lagged versions of macroeconomic variables for the given period
    for var in macro_vars:
        for lag in lags:
            df[f'{var}_lag{lag}'] = df[var].shift(lag)
    
    # Create a DataFrame to store correlation results
    corr_results = pd.DataFrame(index=lags, columns=macro_vars)
    
    # Calculate correlations between Price and lagged variables
    for var in macro_vars:
        for lag in lags:
            lagged_col = f'{var}_lag{lag}'
            corr = df[['Price', lagged_col]].corr().iloc[0, 1]
            corr_results.loc[lag, var] = corr
    
    # Convert values to float for better readability
    corr_results = corr_results.astype(float)
    
    # Display correlation heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_results.T, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title(f'Lagged Correlation of Macroeconomic Indicators with Stock Price ({period_name})')
    plt.xlabel('Lag in Days')
    plt.ylabel('Macroeconomic Indicator')
    plt.tight_layout()
    plt.show()
    
    return corr_results

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Calculate and plot lagged correlations for each period
pre_covid_corr = calculate_lagged_correlations(pre_covid, 'Pre-COVID')
covid_corr = calculate_lagged_correlations(covid, 'COVID')
post_covid_corr = calculate_lagged_correlations(post_covid, 'Post-COVID')


# In[16]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Function to plot time series for 'Price' in each period
def plot_time_series(df, period_name):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Price'], label='Price', color='tab:blue')
    plt.title(f'Time Series of Stock Price ({period_name})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.show()

# Plot time series for each period
plot_time_series(pre_covid, 'Pre-COVID')
plot_time_series(covid, 'COVID')
plot_time_series(post_covid, 'Post-COVID')


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Function to perform seasonal decomposition and plot
def seasonal_decompose_plot(df, period_name, period=12):
    # Set 'Date' as the index for time series analysis
    df.set_index('Date', inplace=True)
    
    # Ensure there are at least two full periods of data
    if len(df) >= 2 * period:
        # Perform seasonal decomposition (using 'Price' column)
        decomposition = seasonal_decompose(df['Price'], model='additive', period=period)  # period=12 for monthly data
        
        # Plot the seasonal decomposition
        plt.figure(figsize=(12, 8))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title(f'Observed ({period_name})')
        
        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title(f'Trend ({period_name})')
        
        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title(f'Seasonal ({period_name})')
        
        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title(f'Residual ({period_name})')
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not enough data for seasonal decomposition in the {period_name} period. Minimum required: {2 * period} data points.")

# Perform seasonal decomposition for each period
seasonal_decompose_plot(pre_covid, 'Pre-COVID', period=12)
seasonal_decompose_plot(covid, 'COVID', period=12)
seasonal_decompose_plot(post_covid, 'Post-COVID', period=12)


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Function to plot histograms for each variable in the dataset
def plot_histograms(df, period_name):
    # Exclude 'Date' column from histogram plotting
    variables = df.columns[df.columns != 'Date']
    
    # Set the number of rows and columns for the subplot grid
    num_vars = len(variables)
    num_cols = 3  # Number of columns in the plot grid
    num_rows = (num_vars // num_cols) + (num_vars % num_cols > 0)  # Calculate number of rows
    
    plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size dynamically
    
    for i, var in enumerate(variables, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.histplot(df[var], kde=True, bins=20, color='skyblue')  # Histogram with KDE (Kernel Density Estimate)
        plt.title(f'{var} Distribution ({period_name})')
        plt.xlabel(var)
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Plot histograms for each period
plot_histograms(pre_covid, 'Pre-COVID')


# In[20]:


plot_histograms(covid, 'COVID')


# In[21]:


plot_histograms(post_covid, 'Post-COVID')


# In[22]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Function to plot scatterplots for 'Price' vs other variables
def plot_scatterplots(df, period_name):
    # Exclude 'Date' column from scatterplot plotting
    variables = df.columns[df.columns != 'Date']
    
    # Set the number of rows and columns for the subplot grid
    num_vars = len(variables)
    num_cols = 3  # Number of columns in the plot grid
    num_rows = (num_vars // num_cols) + (num_vars % num_cols > 0)  # Calculate number of rows
    
    plt.figure(figsize=(15, 5 * num_rows))  # Adjust the figure size dynamically
    
    for i, var in enumerate(variables, 1):
        plt.subplot(num_rows, num_cols, i)
        sns.scatterplot(data=df, x=var, y='Price', color='orange')  # Scatterplot of Price vs other variables
        plt.title(f'{var} vs Price ({period_name})')
        plt.xlabel(var)
        plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()

# Plot scatterplots for each period
plot_scatterplots(pre_covid, 'Pre-COVID')


# In[23]:


plot_scatterplots(covid, 'COVID')


# In[24]:


plot_scatterplots(post_covid, 'Post-COVID')


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Dividing the data into three periods: Pre-COVID, COVID, Post-COVID
pre_covid = df[(df['Date'] >= '2009-07-08') & (df['Date'] <= '2022-06-30')]
covid = df[(df['Date'] >= '2022-07-01') & (df['Date'] <= '2023-12-31')]
post_covid = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-06-30')]

# Function to detect outliers using IQR and plot boxplots
def detect_outliers_iqr(df, period_name):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
    
    # Count the number of outliers
    outlier_count = outliers.shape[0]
    
    # Display boxplot for outliers
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Price', color='lightblue')
    plt.title(f'Price Distribution with Outliers (IQR) - {period_name}')
    plt.xlabel('Price')
    plt.show()
    
    # Return the outlier count
    return outlier_count

# Detect outliers and display boxplots for each period
pre_covid_outliers = detect_outliers_iqr(pre_covid, 'Pre-COVID')
covid_outliers = detect_outliers_iqr(covid, 'COVID')
post_covid_outliers = detect_outliers_iqr(post_covid, 'Post-COVID')

# Display the outlier counts
print(f"Number of outliers in Pre-COVID: {pre_covid_outliers}")
print(f"Number of outliers in COVID: {covid_outliers}")
print(f"Number of outliers in Post-COVID: {post_covid_outliers}")


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Function to detect outliers using IQR and plot boxplot
def detect_outliers_iqr_global(df):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile) for the entire dataset
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    
    # Calculate IQR
    IQR = Q3 - Q1
    
    # Define outlier thresholds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Identify outliers
    outliers = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]
    
    # Count the number of outliers
    outlier_count = outliers.shape[0]
    
    # Display boxplot for outliers
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Price', color='lightblue')
    plt.title('Price Distribution with Outliers (IQR) - Entire Dataset')
    plt.xlabel('Price')
    plt.show()
    
    # Return the outlier count
    return outlier_count

# Detect outliers and display boxplot for the entire dataset
outliers_count = detect_outliers_iqr_global(df)

# Display the count of outliers
print(f"Number of outliers in the entire dataset: {outliers_count}")


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

# --- Load & slice data for each period ---
csv_path = "Final_Stock_prices_and_macroeconomic_data.csv"
df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date").set_index("Date")
df = df.apply(pd.to_numeric, errors="ignore")

# Define the periods
periods = {
    "Pre-COVID":  {"range": ("2009-07-08", "2022-06-30")},
    "COVID":      {"range": ("2022-07-01", "2023-12-31")},
    "Post-COVID": {"range": ("2024-01-01", "2025-06-30")},
}

# --- ADF Test to Check Stationarity ---
def adf_test(series):
    result = adfuller(series, autolag='AIC')  # Perform ADF test
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    print("\n")
    return result[1]  # p-value for stationarity test

# --- Check stationarity for each period ---
for label, cfg in periods.items():
    start, end = cfg["range"]
    print(f"Checking stationarity for {label} period ({start} to {end})")

    # Slice data for the period
    y = df["Price"].loc[start:end].copy()

    # Plot the price data for visualization
    plt.figure(figsize=(10, 5))
    plt.plot(y, label=f"{label} - Price")
    plt.title(f"Price Series for {label} Period")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Perform ADF test to check stationarity
    p_value = adf_test(y)

    # Check if the series is stationary based on the p-value
    if p_value < 0.05:
        print(f"Conclusion: The series is stationary for {label} period.")
    else:
        print(f"Conclusion: The series is non-stationary for {label} period.")


# In[1]:


import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Function to perform ADF test and return p-value
def adf_test(series):
    result = adfuller(series)
    return result[1]  # p-value

# Function to check stationarity
def check_stationarity(df, start_date, end_date, period_name, seasonality=5):
    # Filter the data for the given period
    data_period = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].copy()

    # Set 'Date' as the index for time series analysis
    data_period.set_index('Date', inplace=True)
    
    # Apply seasonal differencing (seasonality = 5 for weekly)
    seasonal_diff = data_period['Price'].diff(seasonality).dropna()
    
    # Apply first differencing
    first_diff = seasonal_diff.diff().dropna()
    
    # Perform ADF test on first differenced data
    p_value = adf_test(first_diff)
    
    # Plot the first differenced series
    plt.figure(figsize=(10, 6))
    plt.plot(first_diff)
    plt.title(f'First Differenced Price - {period_name}')
    plt.show()
    
    print(f"\nADF p-value for {period_name} after seasonal and first differencing: {p_value}")
    
    if p_value < 0.05:
        print(f"The series is stationary after differencing in the {period_name} period.\n")
    else:
        print(f"The series is not stationary after differencing in the {period_name} period.\n")

# Load the dataset
df = pd.read_csv('Final_Stock_prices_and_macroeconomic_data.csv')

# Ensure 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Check stationarity for each period
check_stationarity(df, "2009-07-08", "2022-06-30", "Pre-COVID")
check_stationarity(df, "2022-07-01", "2023-12-31", "COVID")
check_stationarity(df, "2024-01-01", "2025-06-30", "Post-COVID")


# In[ ]:




