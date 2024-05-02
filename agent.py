from langchain_core.prompts import PromptTemplate
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

history = UpstashRedisChatMessageHistory(session_id="chat1", url=os.environ["UPSTASH_URL"],
                                         token=os.environ["UPSTASH_TOKEN"])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)


class InitialRouting(BaseModel):
    routing: str = Field(description="Binary decision, 'web_search' or 'simple_response'")
    search_query: str = Field(
        description="if routing is 'web_search', the text query to search for the information to respond")


initial_routing_parser = PydanticOutputParser(pydantic_object=InitialRouting)

initial_routing_prompt = PromptTemplate(
    template="""
        You are an expert in taking the user input and routing web search or just directly to a simple response.\n
        
        Use the following criteria to decide how to route the input: \n\n
        
        if the user input just require a simple response 
        Just choose 'simple_response' for inputs you can easily answer
        if the response takes more research, choose 'web_search'
        
        Check if the information to answer the question is not already in the chat history.
        
        Chat History: {chat_history}
        
        {format_instructions}
        
        User input to route: {user_input}
        """, input_variables=["user_input", "chat_history"],
    partial_variables={"format_instructions": initial_routing_parser.get_format_instructions()})

initial_routing_chain = initial_routing_prompt | llm_groq_low_T | initial_routing_parser

final_prompt = PromptTemplate(
    template="""
        You are Mark, a personal assistant for Nico, your creator, 
        you an expert in answering user inputs, figure out if the answer needs to be searched via internet,
        or, on the other hand, you can answer it straightaway or with the information present in the chat history\n.
        If there is no context just answer normally.
        
        Chat History: {chat_history}
        
        If the user asks in spanish, answer in spanish, otherwise answer in english.\n
        User input: {user_input} \n
        context:{context}\n
        
        If you dont know the answer just answer 'i dont know'
        """, input_variables=["user_input", "context", "chat_history"])

final_chain = final_prompt | llm_groq | StrOutputParser()

while True:
    user_input = input("You: ")
    memory.chat_memory.add_user_message(user_input)

    if user_input.lower() == 'exit':
        print("Turning off, bye Nico, see you later,")
        break

    res = initial_routing_chain.invoke({"user_input": user_input, "chat_history": history})

    GraphState.initial_routing = res.routing
    GraphState.search_query = res.search_query
    print(f"[DEBUG] routing:{GraphState.initial_routing} , search:{GraphState.search_query}")

    GraphState.search_info = ""
    if GraphState.initial_routing == "web_search":
        search_res = web_search_tool.invoke({"query": GraphState.search_query})
        GraphState.search_info = [x["content"] for x in search_res]
        print(GraphState.search_info)

    res = final_chain.invoke({"user_input": user_input, "context": GraphState.search_info, "chat_history": history})

    memory.chat_memory.add_ai_message(res)
    print(res)
