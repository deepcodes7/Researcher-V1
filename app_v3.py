import os
import pdfplumber
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_index.core import VectorStoreIndex, Settings, Document, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.llms.gemini import Gemini
from llama_index.core.node_parser import SentenceSplitter
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

def extract_text_and_tables(pdf_path):
    """Extract text and tables from a PDF file."""
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract text
            text = page.extract_text()
            if text:
                documents.append(Document(
                    text=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num,
                        "type": "text"
                    }
                ))
            
            # Extract tables
            tables = page.extract_tables()
            for table_num, table in enumerate(tables, 1):
                if table:
                    # Convert table to string representation
                    table_text = "\n".join(["\t".join([str(cell) if cell is not None else "" for cell in row]) for row in table])
                    documents.append(Document(
                        text=f"Table {table_num} from page {page_num}:\n{table_text}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num,
                            "type": "table",
                            "table_number": table_num
                        }
                    ))
    return documents

# Process all PDFs in the data directory
pdf_dir = "data"
all_documents = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_dir, filename)
        documents = extract_text_and_tables(pdf_path)
        all_documents.extend(documents)

# Get list of PDF files
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

# Create a vector index from the documents
index = VectorStoreIndex.from_documents(all_documents)

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
    uvicorn.run("app_v3:app", host="0.0.0.0", port=8000, reload=True) 