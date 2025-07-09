
# 📦 Imports

import streamlit as st
from helper import load_llm, load_prompt

# LangChain - Core Tools
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool

# LangChain - External Tools (Wikipedia, Arxiv)
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper


# 🔧 Load LLM and Prompt

llm = load_llm()
prompt = load_prompt()



# 🧠 Tool 1: Wikipedia Search Tool

wikipedia_wrapper = WikipediaAPIWrapper()
tool1 = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)


# 📚 Tool 2: Arxiv Research Paper Tool

arxiv_wrapper = ArxivAPIWrapper()
tool2 = ArxivQueryRun(api_wrapper=arxiv_wrapper)


# 🌐 Tool 3: VectorDB Retriever Tool

web_docs = WebBaseLoader("https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install").load()

text_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(web_docs)

embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
db = FAISS.from_documents(text_chunks, embedding)
retriever = db.as_retriever()

tool3 = create_retriever_tool(retriever=retriever,
    name="web_search",
    description="Search info on ADK website"
)



# 🤖 Create Agent with Tools

tools = [tool1, tool2, tool3]
agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


# ────────────────────────────────────────────────────────────────────────────────
# 💻 Streamlit Interface
# ────────────────────────────────────────────────────────────────────────────────
st.title("🔍 Multi-Tool AI Agent Demo")

user_query = st.text_input("Enter your prompt:")

if user_query:
    result = executor.invoke({"input": user_query})
    st.write(result)
