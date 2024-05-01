from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
llm = Ollama(model="llama3", temperature=0.7)
output_parser = StrOutputParser()

history = UpstashRedisChatMessageHistory(session_id="chat1", url=os.environ["UPSTASH_URL"],
                                         token=os.environ["UPSTASH_TOKEN"])
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Mark, a personal assistant, Respond to your creator 'Nico' who programmed you"
     "Your objective is to remember things he say and answer any answer he asks or ask you to do"
     "Sometimes he just want to tell you stuff, so you remember, just answer a short answer"
     "Sometimes he will ask you to do something or some questions, then answer with a short answer"
     "if you dont know anything just answer 'I don't know' dont make up things."
     "Dont quote previous conversation if not asked, just make short answers"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Prompt: {input}")])

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=history
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory, output_parser=output_parser)

print("Initializing environment...")
print("Assistant deployed")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    msg = {"input": user_input}
    response = chain.invoke(msg)
    print(response["text"])
