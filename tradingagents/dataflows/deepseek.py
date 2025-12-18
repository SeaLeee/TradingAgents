"""DeepSeek data source for news and fundamentals analysis."""
from openai import OpenAI
import os


def get_deepseek_client():
    """Get DeepSeek client with proper API key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_stock_news_deepseek(query, start_date, end_date):
    """Get stock news analysis using DeepSeek."""
    client = get_deepseek_client()
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a financial news analyst. Provide comprehensive analysis based on your knowledge."
            },
            {
                "role": "user",
                "content": f"""Provide a comprehensive analysis of recent news and events related to {query} 
from {start_date} to {end_date}. Include:
1. Major news events and their market impact
2. Analyst opinions and price targets
3. Industry trends affecting the company
4. Any regulatory or legal developments

Please provide specific details and analysis that would be useful for trading decisions."""
            }
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content


def get_global_news_deepseek(curr_date, look_back_days=7, limit=10):
    """Get global/macro news analysis using DeepSeek."""
    client = get_deepseek_client()
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a macroeconomic analyst specializing in global financial markets."
            },
            {
                "role": "user", 
                "content": f"""Provide a comprehensive analysis of global macroeconomic news and events 
from the past {look_back_days} days leading up to {curr_date}. Include:
1. Central bank policies and interest rate decisions
2. Economic indicators (GDP, inflation, employment)
3. Geopolitical events affecting markets
4. Commodity and currency movements
5. Stock market trends and sector performance

Limit to the {limit} most important developments. Provide analysis that would be useful for trading decisions."""
            }
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content


def get_fundamentals_deepseek(ticker, curr_date):
    """Get fundamental analysis using DeepSeek."""
    client = get_deepseek_client()
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a fundamental equity analyst with expertise in financial statement analysis."
            },
            {
                "role": "user",
                "content": f"""Provide a comprehensive fundamental analysis for {ticker} as of {curr_date}. Include:

1. Key Financial Metrics (format as table):
   - P/E Ratio, P/S Ratio, P/B Ratio
   - EPS (TTM), Revenue Growth, Profit Margins
   - ROE, ROA, ROIC
   - Debt/Equity, Current Ratio
   - Free Cash Flow

2. Business Analysis:
   - Revenue segments and growth drivers
   - Competitive position and moat
   - Management quality and capital allocation

3. Valuation Assessment:
   - Current valuation vs historical
   - Comparison to peers
   - Fair value estimate

4. Risks and Catalysts:
   - Key risks to monitor
   - Upcoming catalysts

Provide specific numbers and analysis based on your knowledge."""
            }
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content
