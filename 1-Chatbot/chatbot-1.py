from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

import streamlit as st


prompy_template=ChatPromptTemplate.from_messages(
    
    '''you are an AI coding assistant. provide only accurate and consise answers to the query.
    
    (query: {query})
    '''
)

st.title("welcome to langchain tutorial")
input_query=st.text_input("Type your question here!!")


LANGUAGE_MODEL = OllamaLLM(model="gemma2:2b")

output_parser = StrOutputParser()

chain = prompy_template | LANGUAGE_MODEL | output_parser

response = chain.invoke( {"query" : input_query})

if input_query:
    st.write(response)