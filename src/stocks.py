import yfinance as yf
from typing import Dict, Optional
from datetime import datetime
from .custom_logger import CustomLogger

class StockTracker:
    """Provide stock data for model analysis."""

    def __init__(self, verbose: bool = False):
        self.logger = CustomLogger.get_logger("Stocks", verbose)
        self.logger.success("StockTracker ready", "STOCKS")
        
        # Known delisted/invalid tickers
        self.delisted_tickers = {
            "YHOO": "Yahoo (YHOO) was delisted in 2017 after being acquired by Verizon. The company now trades as part of Yahoo Japan (TYO: 4689) and Altaba Inc.",
        }

    def get_stock_data(self, query: str) -> Optional[Dict]:
        """Get comprehensive stock data for model."""
        try:
            # Extract ticker from query
            ticker = query.split()[-1].upper()
            
            # Check if ticker is known to be delisted
            if ticker in self.delisted_tickers:
                return {
                    "system": """You have information about a delisted stock. Please explain:
- That the stock is no longer trading
- What happened to the company
- Suggest alternative tickers if available""",
                    "context": {
                        "status": "delisted",
                        "symbol": ticker,
                        "explanation": self.delisted_tickers[ticker],
                        "timestamp": datetime.now().strftime('%B %d, %Y at %I:%M %p %Z')
                    },
                    "query": query
                }
            
            # Get current data for active stocks
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period="1d")
            
            if info and not history.empty:
                return {
                    "system": """You have access to current stock market data. When responding:
- State the current stock price clearly
- Include the company's full name
- Mention when the data was last updated""",
                    "context": {
                        "company": info.get('longName', ''),
                        "symbol": stock.ticker,
                        "current_price": history['Close'].iloc[-1],
                        "open": info.get('regularMarketOpen', 0.0),
                        "high": info.get('regularMarketDayHigh', 0.0),
                        "low": info.get('regularMarketDayLow', 0.0),
                        "volume": info.get('regularMarketVolume', 0),
                        "market_cap": info.get('marketCap', 0),
                        "timestamp": datetime.now().strftime('%B %d, %Y at %I:%M %p %Z')
                    },
                    "query": query
                }
            else:
                return {
                    "system": """You have information about an invalid stock request. Please:
- Explain that the stock symbol couldn't be found
- Suggest checking for typos
- Recommend using valid stock symbols""",
                    "context": {
                        "status": "not_found",
                        "symbol": ticker,
                        "explanation": f"Could not find stock data for symbol: {ticker}",
                        "timestamp": datetime.now().strftime('%B %d, %Y at %I:%M %p %Z')
                    },
                    "query": query
                }
                
        except Exception as e:
            self.logger.info(f"Stock lookup adjustment: {e}", "STOCKS")
            return {
                "system": "You have information about an error in stock data retrieval. Please explain the issue and suggest alternatives.",
                "context": {
                    "status": "error",
                    "symbol": ticker if 'ticker' in locals() else "unknown",
                    "error": str(e),
                    "timestamp": datetime.now().strftime('%B %d, %Y at %I:%M %p %Z')
                },
                "query": query
            }