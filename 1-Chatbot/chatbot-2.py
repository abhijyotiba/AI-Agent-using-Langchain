from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

import os
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT=os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY")


# ✅ API key for OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("❌ OPENROUTER_API_KEY not found. Please set it in your .env file.")


# ✅ Streamlit app
st.title("CODING DOCTOR")
input_query = st.chat_input("How can I help you today?")

# ✅ LLM configuration
LANGUAGE_MODEL = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-small-3.2-24b-instruct"
)

# ✅ Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are an AI coding assistant. Provide only accurate and concise answers to the query.

Query: {query}
"""
)

# ✅ Output parser
OUTPUT_PARSER = StrOutputParser()

# ✅ Chain
CHAIN = prompt_template | LANGUAGE_MODEL | OUTPUT_PARSER

# ✅ Response
if input_query:
    # Show user input as a chat message
    with st.chat_message("user"):
        st.markdown(input_query)

    # Get response from the chain
    response = CHAIN.invoke({"query": input_query})

    # Show the response as AI message
    with st.chat_message("assistant"):
        st.markdown(response)
