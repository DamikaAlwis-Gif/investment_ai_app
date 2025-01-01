from typing import Dict
import yfinance as yf
import pandas as pd
from .errors.finance_exceptions import InvalidStockSymbolError
from .graph_state import GraphState
from .errors.finance_exceptions import MissingStockSymbolError, InsufficientStockSymbolsError
# Tools for different analysis types


def retrieve_single_stock_data(state: GraphState):
    """Comprehensive analysis of a single stock"""
    
    symbol = state["context"]["symbols"][0] if state["context"]["symbols"] else None
    if not symbol:
        raise MissingStockSymbolError()

    period = state["context"].get("period", "1mo")
    if not period:
        period = "1mo"

    stock = yf.Ticker(symbol)
    hist = stock.history(period)
    if hist.empty :
        raise InvalidStockSymbolError(symbol)
      
    return {
        "analysis_results": {
            "historical_data": summarize_stock_data(hist),
            "stock_info": stock.info
        }
    }


def retrieve_multi_stocks_data(state: GraphState):
    """Compare multiple stocks"""

    symbols = state["context"]["symbols"]
    results = {}

    if not symbols or len(symbols) < 2:
        raise InsufficientStockSymbolsError()

    period = state["context"].get("period", "1mo")
    if not period:
        period = "1mo"

    for symbol in symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(period)
        if hist.empty:
            raise InvalidStockSymbolError(symbol)
        results[symbol] = {

            "historical_data": summarize_stock_data(hist),
            "stock_info": stock.info
        }

    return {
        "analysis_results": results
    }


def retrive_technical_analysis_stock_data(state: GraphState):
    """Perform technical analysis on a stock"""

    symbol = state["context"]["symbols"][0] if state["context"]["symbols"] else None
    if not symbol:
        raise MissingStockSymbolError()

    period = state["context"].get("period", "3mo")
    if not period:
        period = "3mo"

    stock = yf.Ticker(symbol)
    hist = stock.history(period)
    if hist.empty:
        raise InvalidStockSymbolError(symbol)
    # Calculate technical indicators
    df = pd.DataFrame(hist)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    return {
        "analysis_results": {

            "historical_data": summarize_stock_data(hist),
            "stock_info": stock.info,
            "trend": "bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "bearish",
            "rsi": df['RSI'].iloc[-1],
            "support": df['Low'].min(),
            "resistance": df['High'].max(),
            "volume_trend": "increasing" if df['Volume'].pct_change().mean() > 0 else "decreasing"
        }
    }

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices (pd.Series): Series of prices
        period (int): The number of periods to use for RSI calculation (default is 14)
    
    Returns:
        pd.Series: RSI values for the given price series
    """
    # Calculate price differences
    delta = prices.diff()
    
    # Separate gains (up) and losses (down)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the specified period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def analyze_sector_performance(sector: str) -> Dict:
    """Analyze performance of a market sector"""
    # You would typically use a sector ETF as proxy
    sector_etfs = {
        "technology": "XLK",
        "healthcare": "XLV",
        "finance": "XLF",
        "energy": "XLE",
        "consumer": "XLY"
    }

    if sector.lower() in sector_etfs:
        etf = yf.Ticker(sector_etfs[sector.lower()])
        hist = etf.history(period="3mo")

        return {
            "current_price": etf.info.get("currentPrice"),
            "monthly_return": hist["Close"].pct_change().mean() * 100,
            "volume_trend": hist["Volume"].pct_change().mean() * 100,
            "top_holdings": etf.info.get("holdings", [])[:5],
            "sector_pe": etf.info.get("forwardPE")
        }
    return {"error": "Sector not found"}


def summarize_stock_data(hist):
    """
    Generate a comprehensive summary of stock historical data
    
    Parameters:
    hist (pd.DataFrame): Historical stock data from yfinance
    
    Returns:
    dict: Summarized stock metrics and analysis
    """
    try:
        # Calculate price changes
        price_change = (hist['Close'].iloc[-1] -
                        hist['Close'].iloc[0]).round(2)
        price_change_pct = (
            (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100).round(2)

        summary = {
            "price_summary": {
                "current_price": round(hist['Close'].iloc[-1], 2),
                "highest_price": round(hist['High'].max(), 2),
                "lowest_price": round(hist['Low'].min(), 2),
                "average_price": round(hist['Close'].mean(), 2),
                "price_change": price_change,
                "price_change_percent": price_change_pct,
            },
            "volume_summary": {
                "average_volume": int(hist['Volume'].mean()),
                "highest_volume": int(hist['Volume'].max()),
                "total_traded_volume": int(hist['Volume'].sum())
            },
            "volatility": {
                # as percentage
                "daily_returns": round(hist['Close'].pct_change().std() * 100, 2),
                "price_std_dev": round(hist['Close'].std(), 2)
            },
            "key_dates": {
                "highest_price_date": hist['High'].idxmax().strftime('%Y-%m-%d'),
                "lowest_price_date": hist['Low'].idxmin().strftime('%Y-%m-%d'),
                "highest_volume_date": hist['Volume'].idxmax().strftime('%Y-%m-%d'),
                "period_start": hist.index[0].strftime('%Y-%m-%d'),
                "period_end": hist.index[-1].strftime('%Y-%m-%d')
            },
            "weekly_summary": {}
        }

        # Calculate weekly summary
        weekly_data = hist['Close'].resample('W').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).tail(4)

        # Convert weekly summary to a more readable format
        for date, values in weekly_data.iterrows():
            summary["weekly_summary"][date.strftime('%Y-%m-%d')] = {
                'open': round(values['open'], 2),
                'close': round(values['close'], 2),
                'high': round(values['high'], 2),
                'low': round(values['low'], 2),
                'volume': int(values['volume'])
            }

        return summary

    except Exception as e:
        raise ValueError(f"Error generating stock summary: {str(e)}")
