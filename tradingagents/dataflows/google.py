from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_global_news_google(
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"] = 7,
    limit: Annotated[int, "max number of articles"] = 10,
) -> str:
    """Get global/macroeconomic news using Google News."""
    # Search for global financial and economic news
    queries = ["stock market", "economy news", "federal reserve", "inflation"]
    
    all_news = []
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before_str = before.strftime("%Y-%m-%d")
    
    for query in queries:
        try:
            news_results = getNewsData(query, before_str, curr_date)
            all_news.extend(news_results[:limit // len(queries)])
        except Exception:
            continue
    
    if not all_news:
        return "No global news found for the specified period."
    
    news_str = ""
    for news in all_news[:limit]:
        news_str += f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
    
    return f"## Global/Macro News from {before_str} to {curr_date}:\n\n{news_str}"