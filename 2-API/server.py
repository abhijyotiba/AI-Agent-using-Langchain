from fastapi import FastAPI , Request
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langserve import add_routes
from dotenv import load_dotenv
import os

from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any


load_dotenv()

# Create FastAPI app with custom OpenAPI settings
app = FastAPI(
    title="LangServe API",
    description="API for blog and poem generation",
    version="1.0.0"
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Define prompt chains
prompt1 = ChatPromptTemplate.from_template(
    "write a clean and concise short story blog on: {topic} , with minimum 200-250 words"
)
prompt2 = ChatPromptTemplate.from_template(
    "write a small poem on: {topic}, keep the poem grounded to the topic"
)

llm1 = ChatOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-small-3.2-24b-instruct",
    temperature=0.9
)

llm2 = OllamaLLM(model="gemma2:2b")

# Define chains
blog_chain: Runnable = prompt1 | llm1
poem_chain: Runnable = prompt2 | llm2


class BlogModel(BaseModel):
    topic : str

BlogModel.model_rebuild()    
    
@app.post("/blog",tags=["API for Blog"])
async def blog_endpoint(request: BlogModel):
    Blog_result = await blog_chain.ainvoke({"topic": request.topic})
    return JSONResponse(content={"result": Blog_result.content})    
    

class poemModel(BaseModel):
    topic : str
poemModel.model_rebuild()
    
@app.post("/poem", tags=["API for poem"])
async def poem_endpoint(request: poemModel):
    Poem_result = await poem_chain.ainvoke({"topic": request.topic})  # also make this async
    return JSONResponse(content={"result": Poem_result})



# Add basic endpoints
@app.get("/")
async def root():
    return {"message": "LangServe API is running", "endpoints": ["/blog", "/poem"]}