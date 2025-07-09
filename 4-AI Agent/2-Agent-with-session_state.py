import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS

from helper import load_llm, load_prompt

from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor

# ------------------- STREAMLIT APP ------------------- #
st.title("🔍 AI Search Agent ")
st.subheader("Tools -[Wikipedia , Arxiv , VectorDB ]")

user_query = st.text_input("Enter your prompt")

# ------------------- SESSION STATE INITIALIZATION ------------------- #
if "retriever" not in st.session_state:
    # Load website and create retriever (Tool 3)
    with st.spinner("Setting up vector store..."):
        web_docs = WebBaseLoader(
            "https://learn.microsoft.com/en-us/windows-hardware/get-started/adk-install"
        ).load()
        text_chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        ).split_documents(web_docs)
        embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
        DB = FAISS.from_documents(text_chunks, embedding)
        st.session_state.retriever = DB.as_retriever()

# ------------------- Load LLM & Prompt ------------------- #
if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

if "prompt" not in st.session_state:
    st.session_state.prompt = load_prompt()

# ------------------- Define Tools ------------------- #
if "tools" not in st.session_state:
    wikipedia_wrapper = WikipediaAPIWrapper()
    tool1 = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

    arxiv_wrapper = ArxivAPIWrapper()
    tool2 = ArxivQueryRun(api_wrapper=arxiv_wrapper)

    tool3 = create_retriever_tool(
        retriever=st.session_state.retriever,
        name="web_search",
        description="search info on web from Microsoft ADK website"
    )

    st.session_state.tools = [tool1, tool2, tool3]

# ------------------- Create Agent + Executor ------------------- #
if "agent_executor" not in st.session_state:
    agent = create_openai_tools_agent(
        st.session_state.llm,
        st.session_state.tools,
        st.session_state.prompt
    )
    st.session_state.agent_executor = AgentExecutor.from_agent_and_tools(
        agent,
        st.session_state.tools,
        verbose=True
    )

# ------------------- Handle Query ------------------- #
if user_query:
    with st.spinner("Thinking..."):
        result = st.session_state.agent_executor.invoke({"input": user_query})
        st.write(result["output"])

