from langchain.agents.format_scratchpad.openai_tools import (format_to_openai_tool_messages)
from langchain.agents import create_openai_functions_agent
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tools import web_search
from langchain_groq import ChatGroq
import os


load_dotenv()
output_parser = StrOutputParser()

tools = [web_search]
# llm = ChatGroq(temperature=0, groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-70b-8192")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

MEMORY_KEY = "chat_history"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are Mark, a personal assistant, Respond to your creator 'Nico' who programmed you"
         "Your objective is to remember things he say and answer any answer he asks or ask you to do"
         "Sometimes he just want to tell you stuff, so you remember, just answer a short answer"
         "Sometimes he will ask you to do something or some questions, then answer with a short answer"
         "if you dont know anything just answer 'I don't know' dont make up things."
         "Use tools to achieve tasks you dont know on your own, line web searching"
         "IMPORTANT: ALWAYS RETURN STRINGS, NOT ARRAYS OR OTHER STRUCTURES, make it human like conversation"
         "Dont quote previous conversation if not asked, just make short answers"),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = ({
             "input": lambda x: x["input"],
             "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                 x["intermediate_steps"]
             ),
             "chat_history": lambda x: x["chat_history"],
         }
         | prompt
         | llm_with_tools
         | OpenAIToolsAgentOutputParser()
         )

history = UpstashRedisChatMessageHistory(session_id="chat1", url=os.environ["UPSTASH_URL"],
                                         token=os.environ["UPSTASH_TOKEN"])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, output_parser=output_parser, verbose=True)

print("Initializing environment...")
print("Assistant deployed")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    msg = {"input": user_input}
    response = agent_executor.invoke(msg)
