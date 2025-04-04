import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from datetime import date
# Application title with style
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>Investment Portfolio Recommendation Dashboard</h1>", unsafe_allow_html=True)

# Cache data
@st.cache_data
def load_data():
    basic_url = "https://docs.google.com/spreadsheets/d/1AyXH3e7GOlN4uyOKYSf3kK0I4BM7Igsn/gviz/tq?tqx=out:csv"
    basic_drive = pd.read_csv(basic_url, index_col=0, usecols=range(8))
    
    sp400_url = "https://docs.google.com/spreadsheets/d/1ALSl_Eh-m8kAruIxDYrHaZoD1z02Dh-g/gviz/tq?tqx=out:csv"
    df_sp400 = pd.read_csv(sp400_url, index_col=0)
    df_sp400.index = pd.to_datetime(df_sp400.index).strftime('%Y-%m-%d')
    
    sp500_url = "https://docs.google.com/spreadsheets/d/1_03xDO9J3kwwvNjiTpiu5njM721o7ZHa/gviz/tq?tqx=out:csv"
    df_sp500 = pd.read_csv(sp500_url, index_col=0)
    df_sp500.index = pd.to_datetime(df_sp500.index).strftime('%Y-%m-%d')
    
    sp600_url = "https://docs.google.com/spreadsheets/d/1myNirpiJJXjyzjDKUM_AW8x4GZH6UJXC/gviz/tq?tqx=out:csv"
    df_sp600 = pd.read_csv(sp600_url, index_col=0)
    df_sp600.index = pd.to_datetime(df_sp600.index).strftime('%Y-%m-%d')
    
    combined_df = pd.concat([df_sp500, df_sp400, df_sp600], axis=1)
    
    # Lấy ngày hiện tại làm giá trị end
    end_date = date.today()

    # Tải dữ liệu từ yfinance với end là ngày hiện tại
    sp500_index = yf.download("^GSPC", start="2019-01-01", end=end_date)
    sp500_close = sp500_index['Close']
    sp400_index = yf.download("^SP400", start="2019-01-01", end=end_date)
    sp400_close = sp400_index['Close']
    sp600_index = yf.download("^SP600", start="2019-01-01", end=end_date)
    sp600_close = sp600_index['Close']
    sp1500_index = yf.download("^SP1500", start="2019-01-01", end=end_date)
    sp1500_close = sp1500_index['Close']
    NYA_index = yf.download("^NYA", start="2019-01-01", end=end_date)
    NYA_close = NYA_index['Close']
    IXIC_index = yf.download("^IXIC", start="2019-01-01", end=end_date)
    IXIC_close = IXIC_index['Close']

    combined_market_df = pd.concat([sp500_close, sp400_close, sp600_close, sp1500_close, NYA_close, IXIC_close], axis=1)
    combined_market_df.columns = ["S&P 500", "S&P 400", "S&P 600", "S&P 1500 Composite", "NYSE Composite", "NASDAQ Composite"]
    
    log_returns = np.log(combined_df / combined_df.shift(1))
    days_in_year = 252
    yearly_return = log_returns.mean() * days_in_year
    yearly_std = log_returns.std() * np.sqrt(days_in_year)
    return basic_drive, combined_df, combined_market_df, yearly_return, yearly_std

basic_drive, combined_df, combined_market_df, yearly_return, yearly_std = load_data()

# Sidebar for risk appetite selection
st.sidebar.markdown("<h3 style='color: #ff7f0e;'>Select Risk Appetite</h3>", unsafe_allow_html=True)
risk_choice = st.sidebar.selectbox("Select preference:", ["Aggressive Investment Style", "Moderate Investment Style", "Conservative Investment Style"])

# Function to filter stocks by risk profile
def filter_stocks_by_risk_profile(risk_profile, df):
    if risk_profile == "Aggressive Investment Style":
        roe_condition = (df['ROE'] > 0.2)
        sector_condition = df['Sector'].isin(['Information Technology', 'Consumer Cyclical', 'Energy', 'Communication Services', 'Real Estate'])
        beta_condition = (df['Beta'] > 1.5)
        pe_condition = (df['P/E'] > 25)
        condition_count = (sector_condition.astype(int) + beta_condition.astype(int) + pe_condition.astype(int))
        conditions = roe_condition & (condition_count >= 2)
    elif risk_profile == "Moderate Investment Style":
        roe_condition = (df['ROE'] > 0.15)
        beta_condition = (df['Beta'].between(0.8, 1.2))
        dividend_condition = (df['Dividend Yield'] > 2)
        pe_condition = (df['P/E'] > 15)
        condition_count = (roe_condition.astype(int) + beta_condition.astype(int) + dividend_condition.astype(int) + pe_condition.astype(int))
        conditions = (condition_count >= 3)
    elif risk_profile == "Conservative Investment Style":
        market_cap_condition = (df['Market Capitalization'] > 10000000000)
        dividend_condition = (df['Dividend Yield'] > 3)
        sector_condition = df['Sector'].isin(['Healthcare', 'Utilities', 'Consumer Defensive'])
        pe_condition = (df['P/E'].between(10, 20))
        roe_condition = (df['ROE'] > 0.15)
        beta_condition = (df['Beta'] < 0.8)
        condition_count = (pe_condition.astype(int) + roe_condition.astype(int) + beta_condition.astype(int))
        conditions = market_cap_condition & dividend_condition & sector_condition & (condition_count >= 2)
    stock_port = df[conditions]
    return stock_port

# Filter stocks
stock_port = filter_stocks_by_risk_profile(risk_choice, basic_drive)
stock_port['Yearly Return'] = yearly_return.loc[stock_port.index]
stock_port['Yearly Std'] = yearly_std.loc[stock_port.index]

# Display stock list
st.markdown(f"<h2 style='color: #2ca02c;'>Stocks Suitable for {risk_choice}</h2>", unsafe_allow_html=True)
st.dataframe(stock_port.style.format({
    'Yearly Return': '{:.4f}',
    'Yearly Std': '{:.4f}',
    'ROE': '{:.4f}',
    'Beta': '{:.4f}',
    'P/E': '{:.4f}',
    'Dividend Yield': '{:.4f}',
    'Market Capitalization': '{:.0f}'
}).background_gradient(cmap='Blues', subset=['Yearly Return', 'Yearly Std']), height=300)

# Fetch risk-free rate data
end_date = date.today()
risk_free_rate_data = yf.download('^TNX', start="2025-01-01", end=end_date)
risk_free_rate = float(risk_free_rate_data['Close'].values[-1] / 100)

# Portfolio optimization
tickers = stock_port.index.tolist()
selected_data = combined_df[tickers]
log_returns_query = np.log(selected_data / selected_data.shift(1)).dropna()

correlation_matrix = log_returns_query.corr()
std_devs = log_returns_query.std()
covariance_matrix = correlation_matrix.to_numpy() * np.outer(std_devs, std_devs)
covariance_df = pd.DataFrame(covariance_matrix, columns=log_returns_query.columns, index=log_returns_query.columns)

def standard_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def expected_return(weights, log_returns_query):
    return np.sum(log_returns_query.mean() * weights) * 252

def sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate):
    return (expected_return(weights, log_returns_query) - risk_free_rate) / standard_deviation(weights, cov_matrix)

def neg_sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate)

stock_list = log_returns_query.columns.tolist()
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
bounds = [(0, 0.4) for _ in range(len(stock_list))]
initial_weights = np.array([1/len(stock_list)] * len(stock_list))

optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns_query, covariance_df, risk_free_rate), 
                             method='SLSQP', constraints=constraints, bounds=bounds)
optimal_weights = optimized_results.x

optimal_portfolio_return = expected_return(optimal_weights, log_returns_query)
optimal_portfolio_volatility = standard_deviation(optimal_weights, covariance_df)
optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns_query, covariance_df, risk_free_rate)

portfolio_df = pd.DataFrame({'Ticker': stock_list, 'Weight': optimal_weights})
portfolio_df = portfolio_df[portfolio_df['Weight'] > 0.0001]

# Display optimal portfolio information
st.markdown("<h2 style='color: #1f77b4;'>Optimal Portfolio Information</h2>", unsafe_allow_html=True)
with st.container():
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Return", f"{optimal_portfolio_return:.2%}")
    col2.metric("Volatility", f"{optimal_portfolio_volatility:.2%}")
    col3.metric("Sharpe Ratio", f"{optimal_sharpe_ratio:.2f}")
    stock_port_opt = stock_port[stock_port.index.isin(portfolio_df['Ticker'].tolist())]
    merged_df = portfolio_df.merge(stock_port_opt[['Dividend Yield']], left_on='Ticker', right_index=True)
    dividend_yield_port = (merged_df['Weight'] * merged_df['Dividend Yield']).sum()
    col4.metric("Dividend Yield", f"{dividend_yield_port:.2f}%")

# Pie chart with Plotly
fig = px.pie(portfolio_df, values='Weight', names='Ticker', title='Optimal Portfolio Weights',
             color_discrete_sequence=px.colors.sequential.Viridis)
fig.update_traces(textinfo='percent+label', pull=[0.1 if w > 0.1 else 0 for w in portfolio_df['Weight']])
st.plotly_chart(fig, use_container_width=True)

# Add correlation matrix between stocks in the optimal portfolio
st.markdown("<h2 style='color: #9467bd;'>Correlation Matrix of Stocks in Portfolio</h2>", unsafe_allow_html=True)

# Filter correlation_matrix for stocks in the optimal portfolio
optimal_tickers = portfolio_df['Ticker'].tolist()
correlation_optimal = correlation_matrix.loc[optimal_tickers, optimal_tickers]

# Create heatmap with Plotly
fig_corr = go.Figure(data=go.Heatmap(
    z=correlation_optimal.values,
    x=correlation_optimal.columns,
    y=correlation_optimal.index,
    colorscale='RdBu',
    zmin=-1, zmax=1,
    text=correlation_optimal.values.round(2),
    texttemplate="%{text}",
    textfont={"size": 10},
    colorbar=dict(title="Correlation")
))

fig_corr.update_layout(
    title="Correlation Matrix of Log-Returns",
    height=600,
    width=800,
    xaxis_title="Stocks",
    yaxis_title="Stocks",
    xaxis=dict(tickangle=45),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)'
)


# Compare Cumulative Returns
portfolio_cumulative = (log_returns_query + 1).cumprod().dot(optimal_weights)
benchmark_returns = np.log(combined_market_df / combined_market_df.shift(1))
benchmark_cumulative = (benchmark_returns + 1).cumprod()
portfolio_cumulative

st.markdown("<h2 style='color: #d62728;'>Cumulative Returns Comparison</h2>", unsafe_allow_html=True)
selected_benchmarks = st.multiselect("Select indices to compare:", list(combined_market_df.columns))
fig = go.Figure()
fig.add_trace(go.Scatter(x=portfolio_cumulative.index, y=portfolio_cumulative, name="Portfolio", line=dict(color="#1f77b4", width=2.5)))
for benchmark in selected_benchmarks:
    fig.add_trace(go.Scatter(x=benchmark_cumulative.index, y=benchmark_cumulative[benchmark], name=benchmark, line=dict(width=2)))
fig.update_layout(title="Cumulative Returns: Portfolio vs Benchmarks", xaxis_title="Date", yaxis_title="Cumulative Return",
                  showlegend=True, height=500, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, use_container_width=True)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.stMetric {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)
