from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory

from dotenv import load_dotenv
from tools import web_search_tool
from langchain_groq import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from graph_state import GraphState

import os

# Chains
# Decide if search
# output result


load_dotenv()

llm_groq_low_T = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-70b-8192", temperature=0.4)
llm_groq = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-70b-8192", temperature=0.4)

history = UpstashRedisChatMessageHistory(session_id="chat2", url=os.environ["UPSTASH_URL"],
                                         token=os.environ["UPSTASH_TOKEN"])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)


class InitialRouting(BaseModel):
    routing: str = Field(description="Binary decision, 'web_search' or 'simple_response'")
    search_query: str = Field(
        description="Full query prompt for a search engine, really especific and detailed to obtain the data")


initial_routing_parser = PydanticOutputParser(pydantic_object=InitialRouting)

initial_routing_prompt = PromptTemplate(
    template="""
        You are an expert in taking the user input and routing web search or just directly to a simple response.\n
        Use the following criteria to decide how to route the input: \n\n
        
        last messages: {chat_history}
        
        If the user asks something, decide if you can answer it on your own, or it has to be searched via internet.
        if the answer is easy, choose 'simple_response' and give no search query
        if research needed choose 'web_search' and build a text query for a search engine to search for that info.
        The text query should include all the words to search, dont omit or refer to context data, name the data explicitly 
                
        {format_instructions}
        
        User input to route: {user_input}
        """, input_variables=["user_input", "chat_history"],
    partial_variables={"format_instructions": initial_routing_parser.get_format_instructions()})

initial_routing_chain = initial_routing_prompt | llm_groq_low_T | initial_routing_parser

final_prompt = ChatPromptTemplate.from_messages([
    ("system", """
        You are Mark, a personal assistant for Nico, your creator,
        you an expert in answering user inputs, figure out if the answer needs to be searched via internet,
        or, on the other hand, you can answer it straightaway or with the information present in the chat history\n.
        If there is no context just answer normally.\n

        If the user asks in spanish, answer in spanish, otherwise answer in english.\n

        Available context: {context}\n

        If you dont know the answer just answer 'i dont know'"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{user_input}")])

final_chain = final_prompt | llm_groq | StrOutputParser()

while True:
    user_input = input("You: ")
    memory.chat_memory.add_user_message(user_input)

    if user_input.lower() == 'exit':
        print("Turning off, bye Nico, see you later,")
        break

    chat_history = memory.chat_memory.messages
    res = initial_routing_chain.invoke({"user_input": user_input, "chat_history": chat_history[::-1][0:15]})

    GraphState.initial_routing = res.routing
    GraphState.search_query = res.search_query
    print(f"[DEBUG] routing:{GraphState.initial_routing} , search:{GraphState.search_query}")

    GraphState.search_info = ""
    if GraphState.initial_routing == "web_search":
        search_res = web_search_tool.invoke({"query": GraphState.search_query})
        GraphState.search_info = [x["content"] for x in search_res]
        print(GraphState.search_info)

    res = final_chain.invoke(
        {"user_input": user_input, "context": GraphState.search_info, "chat_history": chat_history})

    print(chat_history[::-1][0:6])
    memory.chat_memory.add_ai_message(res)
    print(res)
