from typing_extensions import TypedDict
from typing import List


class GraphState(TypedDict):
    """"
    State of our graph

    initial_routing: the initial route we took, either erb_search or simple response
    search_query: Query for the web search
    search_info: What the web search gathered
    final_answer: Final answer to the user
    """
    initial_routing: str
    search_query: str
    search_info: List[str]
    final_answer: str
    num_steps: int
