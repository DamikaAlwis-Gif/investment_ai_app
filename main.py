import yfinance as yf
import requests
from typing import TypedDict, List, Dict, Any, Literal
from bs4 import BeautifulSoup


def get_market_news(keywords: List[str]) -> List[Dict]:
    """Fetch relevant market news based on keywords"""
    url = "https://finance.yahoo.com/news"
    headers = {'User-Agent': 'Mozilla/5.0'}

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []

        for article in soup.select('div.js-stream-content'):
            title = article.select_one('h3')
            if title and any(keyword.lower() in title.text.lower() for keyword in keywords):
                news_items.append({
                    'title': title.text,
                    'date': datetime.now().strftime("%Y-%m-%d")
                })

        return news_items[:5]
    except Exception as e:
        return [{"error": str(e)}]
def main():
  stock = yf.Ticker("AAPL")
  hist = stock.history(period="1mo")
  print(hist)
  print(stock.info)
#   news = stock.get_news(4)
#   print(news)

if __name__ == "__main__":
  main()