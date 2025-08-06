# To run this application:
# 1. Make sure you have a .env file with your service keys (e.g., GOOGLE_API_KEY, SUPABASE_URL, TOGETHER_API_KEY, SUPABASE_SERVICE_KEY).
# 2. Ensure your Supabase database is set up with the 'vector' extension.
# 3. Install required packages: pip install -r requirements.txt
# 4. Run the server: python -m uvicorn main:app --reload

import os
import tempfile
import json
import requests
import asyncio
import traceback
from dotenv import load_dotenv
from typing import List, Optional

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

# --- Langchain and related imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_together import ChatTogether
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Supabase Imports ---
from supabase.client import Client, create_client

# --- 1. INITIALIZATION & CONFIGURATION ---

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Document Q&A API",
    description="An API to find answers within documents (PDF, DOCX, EML). It ingests documents from a URL, stores them in a persistent vector database, and allows you to ask multiple questions.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Lazy Loading for Models and Clients ---
llm: Optional[ChatTogether] = None
gemini_embedder: Optional[GoogleGenerativeAIEmbeddings] = None
supabase_client: Optional[Client] = None

# --- 2. STARTUP EVENT ---

@app.on_event("startup")
async def startup_event():
    """On application startup, initialize components."""
    print("Application startup: Initializing components...")
    initialize_components()

def initialize_components():
    """Initializes models and clients if they haven't been already."""
    global llm, gemini_embedder, supabase_client
    if llm is None:
        print("Initializing Together AI LLM...")
        llm = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct-v0.1", temperature=0.1)
    if gemini_embedder is None:
        print("Initializing Google Gemini Embedding model...")
        gemini_embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    if supabase_client is None and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        print("Initializing Supabase client...")
        supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print("Models and clients are ready.")


# --- Prompt Template for the LLM ---
PROMPT_TEMPLATE = PromptTemplate(
    template="""
    **Your Role:** You are a helpful AI assistant. Your task is to answer the user's question based *only* on the provided document context.
    **Strict Constraints:**
    - Your answer **MUST** be derived solely from the `Document Context`. Do not use any external knowledge.
    - If the answer is not in the context, state: "The answer to this question could not be found in the provided document."
    ---
    **Document Context:** {context}
    ---
    **User Query:** {question}
    ---
    **Helpful Answer:**
    """,
    input_variables=["question", "context"]
)


# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    document_url: HttpUrl = Field(..., description="A single public URL to a document (PDF, DOCX, EML).")
    questions: List[str] = Field(..., min_length=1, description="A non-empty list of questions to ask about the document.")


class QueryResponse(BaseModel):
    answers: List[str]
    document_url: HttpUrl
    message: str


# --- 3. CORE API ENDPOINT ---
@app.post('/query', response_model=QueryResponse, tags=["Document Q&A"])
async def query_document(payload: QueryRequest):
    """
    This endpoint ingests a document if new, then finds answers to questions within it.
    - If a document has been processed before, it uses the cached version from the vector store.
    - Otherwise, it downloads, processes, and stores the document for future queries.
    """
    print(f"\n--- New Request Received for /query ---")

    if not supabase_client:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database client is not available. Check Supabase credentials.")

    doc_url = str(payload.document_url)
    questions = payload.questions
    print(f"Processing document from URL: {doc_url}")
    print(f"Answering {len(questions)} questions.")

    vectorstore = None
    ingestion_message = ""

    try:
        # --- Check for Existing Embeddings in Supabase ---
        print("Step 1: Checking for existing document vectors in Supabase...")
        response = supabase_client.from_("documents").select("id", count='exact').eq("metadata->>source", doc_url).limit(1).execute()

        if response.count > 0:
            print("DATABASE HIT: Found pre-processed vectors. Skipping ingestion.")
            ingestion_message = "Document already processed. Using existing vectors from database."
            vectorstore = SupabaseVectorStore(
                client=supabase_client,
                embedding=gemini_embedder,
                table_name="documents",
                query_name="match_documents"
            )
        else:
            print("DATABASE MISS: Processing and embedding new document.")
            ingestion_message = "New document processed and vectors stored in database."
            temp_file_path = None
            try:
                print("Step 1a: Downloading document...")
                http_response = requests.get(doc_url)
                http_response.raise_for_status()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                    temp_file.write(http_response.content)
                    temp_file_path = temp_file.name

                lower_doc_url = doc_url.lower()
                if lower_doc_url.endswith('.pdf'): loader = PyPDFLoader(temp_file_path)
                elif lower_doc_url.endswith('.docx'): loader = Docx2txtLoader(temp_file_path)
                elif lower_doc_url.endswith('.eml'): loader = UnstructuredEmailLoader(temp_file_path)
                else: loader = PyPDFLoader(temp_file_path)

                pages = loader.load()
                if not pages: raise ValueError("Could not load content from the document.")
                print(f"Step 2: Loaded {len(pages)} pages/sections.")

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(pages)
                if not docs: raise ValueError("Document could not be split into processable chunks.")

                for doc in docs:
                    doc.metadata = {"source": doc_url}

                print(f"Step 3: Split document into {len(docs)} chunks. Uploading to Supabase...")
                vectorstore = await SupabaseVectorStore.afrom_documents(
                    documents=docs,
                    embedding=gemini_embedder,
                    client=supabase_client,
                    table_name="documents",
                    query_name="match_documents",
                    chunk_size=50
                )
                print("Step 4: Supabase vector store created successfully.")

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"Temporary file removed: {temp_file_path}")

        # --- Create Retriever and QA Chain ---
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15, "filter": {"source": doc_url}})
        print("Step 5: Initialized retriever with source document filter.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
        )
        print("Step 6: RAG QA Chain created.")

    except Exception as e:
        print(f"ERROR during document processing or vector store setup: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to process document: {str(e)}")

    # --- Question Answering ---
    try:
        async def get_answer(chain, query):
            print(f"Processing query: '{query}'")
            try:
                result = await chain.ainvoke({"query": query})
                answer = result.get('result', 'Error: Could not process this question.').strip()
                print(f"Successfully answered: '{query}'")
                return answer
            except Exception as e:
                error_message = f"Error for query '{query}': {str(e)}"
                print(f"ERROR invoking chain: {error_message}")
                traceback.print_exc()
                return error_message

        print(f"Step 7: Starting parallel processing for {len(questions)} questions...")
        tasks = [get_answer(qa_chain, q) for q in questions]
        answers = await asyncio.gather(*tasks)
        print("All questions processed successfully.")

        final_response = {
            "answers": answers,
            "document_url": doc_url,
            "message": ingestion_message
        }
        print(f"Final response prepared: {json.dumps(final_response, indent=2)}")
        return final_response

    except Exception as e:
        print(f"ERROR during question answering: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An error occurred during question answering: {str(e)}")


# --- Health Check Endpoint ---
@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok", "name": "Intelligent Document Q&A API"}
