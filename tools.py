from langchain.agents import tool
from dotenv import load_dotenv
from langchain_community.tools import BingSearchRun
from langchain_community.utilities import BingSearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return 123


web_search_tool = TavilySearchResults(k=2)
#
# api_wrapper = BingSearchAPIWrapper()
# bing_search = BingSearchRun(api_wrapper=api_wrapper)
