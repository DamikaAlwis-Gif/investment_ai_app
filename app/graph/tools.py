from typing import TypedDict, List, Dict, Any, Literal
import yfinance as yf
import pandas as pd
from .errors.finance_exceptions import InvalidStockSymbolError
# Tools for different analysis types

def analyze_single_stock(symbol: str) -> Dict:
    """Comprehensive analysis of a single stock"""
    
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1mo")
    if hist.empty :
        raise InvalidStockSymbolError(symbol)
    
    return {
        "current_price": stock.info.get("currentPrice"),
        "price_change": hist["Close"].pct_change().mean(),
        "volume": stock.info.get("volume"),
        "pe_ratio": stock.info.get("forwardPE"),
        "market_cap": stock.info.get("marketCap"),
        "52w_high": stock.info.get("fiftyTwoWeekHigh"),
        "52w_low": stock.info.get("fiftyTwoWeekLow")
    }


def compare_stocks(symbols: List[str]) -> Dict:
    """Compare multiple stocks"""
    results = {}
    for symbol in symbols:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1mo")
        if hist.empty:
            raise InvalidStockSymbolError(symbol)
        results[symbol] = {
            "current_price": stock.info.get("currentPrice"),
            "monthly_return": hist["Close"].pct_change().mean(),
            "market_cap": stock.info.get("marketCap"),
            "pe_ratio": stock.info.get("forwardPE")
        }
    return results


def technical_analysis(symbol: str) -> Dict:
    """Perform technical analysis on a stock"""
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo")
    if hist.empty:
        raise InvalidStockSymbolError(symbol)
    # Calculate technical indicators
    df = pd.DataFrame(hist)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    
    return {
        "trend": "bullish" if df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else "bearish",
        "rsi": df['RSI'].iloc[-1],
        "support": df['Low'].min(),
        "resistance": df['High'].max(),
        "volume_trend": "increasing" if df['Volume'].pct_change().mean() > 0 else "decreasing"
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