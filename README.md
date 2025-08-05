# Intelligent Document Q&A API

An asynchronous, high-performance RAG (Retrieval-Augmented Generation) API designed to answer questions about various documents (PDFs, DOCX files, EMLs) from a URL with speed and accuracy. This project is engineered for stability, low-latency responses, and reliable operation on cloud platforms.

---

## 🔑 Key Features

- **Multi-Format Document Support**  
  Intelligently processes different document types including `.pdf`, `.docx`, and `.eml` by automatically selecting the appropriate document loader.

- **Hybrid AI Model Integration**  
  Leverages Google's `embedding-001` for robust and accurate document embedding, combined with Together AI's high-speed `Qwen/Qwen3-235B` model for fast and intelligent question answering.

- **High-Performance Caching**  
  Implements a memory-aware, in-memory cache. The first request for a new document URL is processed and cached; all subsequent requests for the same document are nearly instantaneous.

- **Asynchronous Architecture**  
  Built on **FastAPI** and **Uvicorn**, the application is fully asynchronous, capable of handling multiple concurrent requests without blocking.

- **Optimized for Cloud Deployment**  
  Includes a background keep-alive task and lazy loading of models to ensure reliable, continuous operation on free-tier hosting platforms like **Render**.

- **Robust Concurrency Control**  
  Uses an `asyncio.Semaphore` to manage the flow of API calls to the LLM, preventing rate-limit errors and ensuring stability under load.

- **Output Post-Processing**  
  Includes logic to filter and clean the LLM's output, guaranteeing a clean, human-readable JSON response without any extraneous thought processes.

---

## 🧱 Tech Stack

- **Backend**: FastAPI, Python 3  
- **AI & Machine Learning**: LangChain, Together AI, Google Gemini, FAISS  
- **Deployment**: Render, Uvicorn  
- **Core Libraries**: Pydantic, httpx, PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader  

---

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### ✅ Prerequisites

- Python 3.11+
- A package manager like `pip`

### 1. Clone the Repository

```bash
git clone https://github.com/hacketthadwin/Intelligent-Document-QnA-API.git
cd Intelligent-Document-QnA-API
```

### 2. Set Up Environment Variables

Create a file named `.env` in the root of the project directory and add your API keys and a security token:

```env
GOOGLE_API_KEY="your_google_api_key_here"
TOGETHER_API_KEY="your_together_ai_api_key_here"
API_AUTH_TOKEN="a_long_random_secret_string_of_your_choice"
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the development server using **Uvicorn**:

```bash
python -m uvicorn app:app --reload
```

The API will now be running on [http://127.0.0.1:8000](http://127.0.0.1:8000)

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 📡 API Endpoint

### `POST /api/process`

This endpoint downloads a document from a given URL, processes it, and answers a list of questions based on its content.

#### Headers

```http
Authorization: Bearer YOUR_API_AUTH_TOKEN
Content-Type: application/json
```

#### Request Body (JSON)

```json
{
  "documents": "https://your-publicly-accessible-document-url.docx",
  "questions": [
    "What is the grace period for premium payment?",
    "What is the waiting period for pre-existing diseases?",
    "Does this policy cover maternity expenses?"
  ]
}
```

#### Successful Response (200 OK)

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date.",
    "There is a waiting period of thirty-six (36) months for pre-existing diseases to be covered.",
    "Yes, the policy covers maternity expenses after a waiting period of 24 months."
  ]
}
```

---

## 🌐 Deployment

This application is configured for easy deployment on **Render**.

### 🔧 Build & Start Commands

- **Build Command**:  
  ```bash
  pip install -r requirements.txt
  ```

- **Start Command**:  
  ```bash
  uvicorn app:app --host 0.0.0.0 --port $PORT
  ```

---

