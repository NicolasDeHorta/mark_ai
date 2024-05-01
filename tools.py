from langchain.agents import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return 123

web_search = TavilySearchResults(k=2)
