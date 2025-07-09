import os
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


def load_llm():
    
    llm = ChatOpenAI(
        api_key= os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="mistralai/mistral-small-3.2-24b-instruct"
    )
    return llm

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

def load_prompt():
    prompt= ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are an AI assistant with tools integrated to answer the query. "
            "Use the tools provided to fetch relevant information before giving a generalized answer. "
            "Make sure the response is accurate and backed by facts and relevant sources."
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")  # 👈 required for tool use
    ])
    return prompt
