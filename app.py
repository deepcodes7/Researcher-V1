import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set API keys
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Model configuration
MODELS = {
    "gpt-4": {
        "name": "GPT-4",
        "llm": OpenAI(model="gpt-4-turbo", request_timeout=60.0)
    },
    "gemini": {
        "name": "Gemini",
        "llm": Gemini(model="gemini-2.0-flash", api_key=GOOGLE_API_KEY)
    }
}

# Initialize the default model
DEFAULT_MODEL = "gpt-4"
Settings.llm = MODELS[DEFAULT_MODEL]["llm"]
Settings.transformations = [SentenceSplitter(chunk_size=512, chunk_overlap=100)]

# Load PDF file(s) from the "data" directory
pdf_reader = SimpleDirectoryReader(input_dir="data", required_exts=[".pdf"])
documents = pdf_reader.load_data()

# Get list of PDF files
pdf_files = [os.path.basename(f) for f in os.listdir("data") if f.endswith(".pdf")]

# Create a vector index from the documents
index = VectorStoreIndex.from_documents(documents)

# Create a query engine for Retrieval-Augmented Generation (RAG)
query_engine = index.as_query_engine()

# Initialize FastAPI app
app = FastAPI(title="AI Document Search")

# Set up templates
templates = Jinja2Templates(directory="templates")

class Query(BaseModel):
    query: str
    model: str = DEFAULT_MODEL

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": {k: v["name"] for k, v in MODELS.items()},
        "default_model": DEFAULT_MODEL,
        "pdfs": pdf_files
    })

@app.post("/query")
async def query_pdf(query: Query):
    # Update the model if it exists in our configuration
    if query.model in MODELS:
        Settings.llm = MODELS[query.model]["llm"]
        print(f"Using model: {query.model}")
        global query_engine
        query_engine = index.as_query_engine()
    
    response = query_engine.query(query.query)
    print(f"Response: {response}")
    return {"response": str(response)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 