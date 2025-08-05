# app.py
# FINAL PRODUCT VERSION - Generalized for use as a standalone service.
# To run this application:
# 1. Make sure you have a .env file with your GOOGLE_API_KEY, TOGETHER_API_KEY and API_AUTH_TOKEN.
# 2. Install the required packages from requirements.txt.
# 3. Run the server from your terminal:
#    python -m uvicorn app:app --reload

import os
import tempfile
import json
import requests
import asyncio
import traceback
from dotenv import load_dotenv
from typing import List, Optional, Union, Dict

# For the keep-alive task
import httpx

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, HTTPException, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

# --- Langchain and related imports ---
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_together import ChatTogether
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- 1. INITIALIZATION & CONFIGURATION ---

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") # For embeddings
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY") # For chat model
API_AUTH_TOKEN = os.environ.get("API_AUTH_TOKEN") # Generic auth token
RENDER_EXTERNAL_URL = os.environ.get("RENDER_EXTERNAL_URL")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. Will rely on environment variables.")
if not TOGETHER_API_KEY:
    print("Warning: TOGETHER_API_KEY not found. Will rely on environment variables.")
if not API_AUTH_TOKEN:
    print("Warning: API_AUTH_TOKEN not found. Will rely on environment variables.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Document Q&A API",
    description="A highly reliable, asynchronous API to find answers within various documents (PDF, DOCX, EML) from a URL.",
    version="1.0.0" # Product Version
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Caching Mechanism with Memory Limit ---
QA_CHAIN_CACHE: Dict[str, RetrievalQA] = {}
MAX_CACHE_SIZE = 1  # Only keep the most recent document in memory to save RAM

# --- Lazy Loading for LLM components ---
llm: Optional[ChatTogether] = None
gemini_embedder: Optional[GoogleGenerativeAIEmbeddings] = None

# --- 2. BACKGROUND KEEP-ALIVE TASK ---

KEEP_ALIVE_INTERVAL_SECONDS = 14 * 60  # 14 minutes

async def keep_alive_task():
    """A background task that pings the server to keep it from spinning down."""
    await asyncio.sleep(30)
    print("--- Starting keep-alive task. ---")
    while True:
        await asyncio.sleep(KEEP_ALIVE_INTERVAL_SECONDS)
        if RENDER_EXTERNAL_URL:
            ping_url = f"{RENDER_EXTERNAL_URL}/health"
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(ping_url, timeout=20)
                print(f"Keep-alive ping sent to {ping_url}. Status: {response.status_code}")
            except Exception as e:
                print(f"Keep-alive ping failed: {e}")
        else:
            print("Skipping keep-alive ping (RENDER_EXTERNAL_URL not set).")

@app.on_event("startup")
async def startup_event():
    """On application startup, create the background keep-alive task."""
    print("Application startup: Initializing keep-alive background task...")
    asyncio.create_task(keep_alive_task())

def initialize_llm_components():
    """Initializes the LLM and embedding models if they haven't been already."""
    global llm, gemini_embedder
    if llm is None:
        print("Initializing Together AI LLM for the first time...")
        llm = ChatTogether(
            model="Qwen/Qwen3-235B-A22B-Thinking-2507",
            temperature=0.1
        )
    if gemini_embedder is None:
        print("Initializing Google Gemini Embedding model for the first time...")
        gemini_embedder = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    print("LLM and Embedding models are ready.")


# --- Prompt Template for the LLM ---
PROMPT_TEMPLATE = PromptTemplate(
    template="""
    **Your Role:** You are a helpful AI assistant. Your task is to answer the user's question based *only* on the provided document context.

    **Core Task:**
    1.  Read the `User Query` and the `Document Context` carefully.
    2.  Find the specific information in the context that answers the question.
    3.  Synthesize this information into a clear, concise, and natural-sounding answer.
    4.  Phrase the answer as a complete sentence or a helpful statement. Do not output raw text fragments.

    **Example:**
    -   **User Query:** "What is the waiting period for cataracts?"
    -   **Document Context:** "...iii. Two years waiting period a. Cataract..."
    -   **Good Answer:** "The policy has a specific waiting period of two (2) years for cataract surgery."
    -   **Bad Answer:** "Two years waiting period a. Cataract"

    **Strict Constraints:**
    -   Your answer **MUST** be derived solely from the `Document Context`. Do not use any external knowledge.
    -   If the answer is not in the context, state: "The answer to this question could not be found in the provided document."

    ---
    **Document Context:**
    {context}
    ---
    **User Query:**
    {question}
    ---
    **Helpful Answer:**
    """,
    input_variables=["question", "context"]
)

# --- Pydantic Models for Request and Response (Data Validation) ---
class ProcessRequest(BaseModel):
    documents: Union[List[HttpUrl], HttpUrl] = Field(..., description="A single URL or a list containing a single URL to a document.")
    questions: List[str] = Field(..., min_length=1, description="A non-empty list of questions to ask about the document.")

class ProcessResponse(BaseModel):
    answers: List[str]

# --- 3. CORE API ENDPOINT ---

@app.post('/api/process', response_model=ProcessResponse, tags=["Document Processing"])
async def process_document(
    payload: ProcessRequest,
    Authorization: Optional[str] = Header(None, description="Bearer token for authentication.")
):
    """
    This endpoint performs the entire RAG process for a given document URL and a list of questions.
    It uses a memory-aware cache to provide low-latency responses for the most recent document.
    """
    print("\n--- New Request Received for /api/process ---")

    # --- 3a. Authentication ---
    expected_header = f"Bearer {API_AUTH_TOKEN}"
    if not Authorization or Authorization != expected_header:
        print(f"Authentication failed. Received: {Authorization}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed or token is missing.",
        )
    print("Authentication successful.")

    # --- 3b. Input Normalization ---
    if not isinstance(payload.documents, list):
        documents_list = [payload.documents]
    else:
        documents_list = payload.documents

    if not documents_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The 'documents' field cannot be an empty list.",
        )

    doc_url = str(documents_list[0])
    questions = payload.questions
    print(f"Processing document from URL: {doc_url}")
    print(f"Answering {len(questions)} questions.")

    qa_chain = None

    # --- 3c. Cache Check ---
    if doc_url in QA_CHAIN_CACHE:
        print("CACHE HIT: Found pre-processed QA chain. Skipping document processing.")
        qa_chain = QA_CHAIN_CACHE[doc_url]
    else:
        print("CACHE MISS: Processing new document.")
        
        if len(QA_CHAIN_CACHE) >= MAX_CACHE_SIZE:
            oldest_key = next(iter(QA_CHAIN_CACHE))
            print(f"Cache full. Removing oldest item: {oldest_key}")
            del QA_CHAIN_CACHE[oldest_key]

        temp_file_path = None
        try:
            initialize_llm_components()

            print("Step 1: Downloading document...")
            response = requests.get(doc_url)
            response.raise_for_status()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
                temp_file.write(response.content)
                temp_file_path = temp_file.name
            
            print(f"Step 1b: Detecting document type from URL...")
            lower_doc_url = doc_url.lower()
            if lower_doc_url.endswith('.pdf'):
                print("PDF document detected.")
                loader = PyPDFLoader(temp_file_path)
            elif lower_doc_url.endswith('.docx'):
                print("DOCX document detected.")
                loader = Docx2txtLoader(temp_file_path)
            elif lower_doc_url.endswith('.eml'):
                print("EML (email) document detected.")
                loader = UnstructuredEmailLoader(temp_file_path)
            else:
                print("Unknown document type. Attempting to load as PDF as a fallback.")
                loader = PyPDFLoader(temp_file_path)

            pages = loader.load()
            if not pages:
                raise ValueError("Could not load any content from the document.")
            print(f"Step 2: Loaded {len(pages)} pages/sections from the document.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(pages)
            print(f"Step 3: Split document into {len(docs)} chunks.")

            print("Step 4: Generating embeddings and creating vector store...")
            vectorstore = await FAISS.afrom_documents(docs, embedding=gemini_embedder)
            print("Step 5: FAISS vectorstore created successfully.")

            retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
            print("Step 6: Initialized high-recall retriever.")
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": PROMPT_TEMPLATE}
            )
            print("Step 7: RAG QA Chain created. Storing in cache...")
            QA_CHAIN_CACHE[doc_url] = qa_chain

        except Exception as e:
            print(f"ERROR during document processing: {e}")
            traceback.print_exc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process document: {str(e)}",
            )
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                print(f"Temporary file removed: {temp_file_path}")

    # --- 3d. Question Answering ---
    try:
        semaphore = asyncio.Semaphore(10)

        async def get_answer(chain, query):
            async with semaphore:
                print(f"Processing query: '{query}'")
                try:
                    result = await chain.ainvoke({"query": query})
                    raw_answer = result.get('result', 'Error: Could not process this question.').strip()
                    
                    if "</think>" in raw_answer:
                        clean_answer = raw_answer.split("</think>", 1)[-1].strip()
                        print(f"Successfully answered (filtered): '{query}'")
                        return clean_answer
                    else:
                        print(f"Successfully answered (no filter needed): '{query}'")
                        return raw_answer

                except Exception as e:
                    error_message = f"Error for query '{query}': {str(e)}"
                    print(f"ERROR invoking chain: {error_message}")
                    traceback.print_exc()
                    return error_message

        print(f"Step 8: Starting high-speed parallel processing for {len(questions)} questions...")
        tasks = [get_answer(qa_chain, q) for q in questions]
        answers = await asyncio.gather(*tasks)
        print("All questions processed successfully.")

        final_response = {"answers": answers}
        print(f"Final response prepared: {json.dumps(final_response, indent=2)}")
        return final_response

    except Exception as e:
        print(f"ERROR during question answering: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An internal server error occurred during question answering: {str(e)}",
        )

# --- Health Check Endpoint (used by the keep-alive task) ---
@app.get("/health", tags=["General"])
def health_check():
    """A simple health check endpoint to confirm the API is running."""
    return {"status": "ok"}
