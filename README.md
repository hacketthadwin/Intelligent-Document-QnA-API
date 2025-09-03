
# Intelligent Document Q&A API
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://hacketthadwin-intelligent-document-qna-api.hf.space)
[![Status](https://img.shields.io/badge/status-complete-brightgreen)](#)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![Framework](https://img.shields.io/badge/framework-FastAPI-blue)](#)
[![LangChain](https://img.shields.io/badge/LangChain-enabled-yellow)](#)
[![Database](https://img.shields.io/badge/db-Supabase%20%2B%20pgvector-009688)](#)
[![Model](https://img.shields.io/badge/AI-Google%20Gemini-orange)](#)

A high-performance API that allows you to find answers within your documents using powerful language models. Ingest PDFs, DOCX files, or emails, and get back precise answers to your questions.

This project is designed to be simple to set up and use, acting as a robust backend for any application that needs document-based question-answering capabilities.

---

Live API Demo: https://hacketthadwin-intelligent-document-qna-api.hf.space

---
### 🚀 What Makes This Project Different?

Unlike many existing RAG APIs, this project is:

- **Model-Agnostic and Multi-Provider Friendly**  
  Supports **Google Gemini** - no hard dependency on OpenAI.

- **Cloud-Ready and Free-Tier Optimized**  
  Specifically engineered to run smoothly on platforms like **Hugging Face Spaces**, with memory-efficient caching and eager model loading.

- **Format-Intelligent**  
  Automatically detects and uses the correct loader for `.pdf`, `.docx`, and `.eml` files — no manual preprocessing required.

- **Minimal Memory Footprint**  
  Designed for low-resource environments — ideal for free-tier deployments, research prototypes, or student projects.

- **Clean JSON Output**  
  Filters out verbose LLM reasoning ("Thought: Let's find the answer...") and returns only the clean, relevant answers.

This makes it ideal for developers, students, and startups looking to build document Q&A apps without the complexity or cost of large RAG systems.

---

## ✨ Features

- **Multi-Format Support**  
  Natively handles `.pdf`, `.docx`, and `.eml` files.

- **Persistent Storage**  
  Uses **Supabase** with `pgvector` to store document embeddings. Process a document once and query it instantly anytime after.

- **High-Quality Answers**  
  Leverages state-of-the-art language models from **Gemini AI** and **Google** for accurate embeddings and intelligent Q&A.

- **Asynchronous & Fast**  
  Built with **FastAPI** for high-performance, non-blocking I/O.

- **Easy to Deploy**  
  Ready to be containerized with **Docker** or deployed to any modern cloud platform.


---

## 🚀 Getting Started

Follow these steps to get the API server running on your local machine.

### ✅ Prerequisites

- Python 3.8+
- A Supabase account with a project created
- API keys from Google AI Studio and Gemini AI

---

### 📁 1. Clone the Repository

```bash
git clone https://github.com/hacketthadwin/intelligent-document-qna-api.git
cd intelligent-document-qna-api
```

---

### 🗃️ 2. Set Up Your Supabase Database

Enable the `vector` extension in your Supabase project:

1. Go to your Supabase project dashboard  
2. Navigate to the **SQL Editor**  
3. Run the following SQL command:

```sql
create extension if not exists vector;
```

A `documents` table will be automatically created the first time a document is processed via LangChain.

---

### 🔐 3. Configure Environment Variables

Create a `.env` file in the root of your project directory. Use this template:

```env
# --- Service Keys ---
GOOGLE_API_KEY="your_google_api_key_here"
# --- Supabase Credentials for Vector Store ---
SUPABASE_URL="https://your_supabase_project_id.supabase.co"
SUPABASE_SERVICE_KEY="your_supabase_service_role_key_here"
```

---

### 📦 4. Install Dependencies

Install required packages using pip:

```bash
pip install -r requirements.txt
```

---

### ▶️ 5. Run the API Server

Start the FastAPI server with:

```bash
python -m uvicorn main:app --reload
```

Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the API.  
Interactive docs available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## ⚙️ API Usage

### `POST /query`

This endpoint ingests a document (if it's new) and answers questions about it.

---

### 📤 Request Body

- `document_url` (string, required): Public URL to the document you want to query  
- `questions` (array of strings, required): One or more questions to ask

#### ✅ Example using `curl`

```bash
curl -X POST "http://127.0.0.1:8000/query" \
-H "Content-Type: application/json" \
-d '{
  "document_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "questions": [
    "What is the title of this paper?",
    "Summarize the abstract in one sentence."
  ]
}'
```

---

### 📥 Example Success Response

```json
{
  "answers": [
    "The title of the paper is 'Attention Is All You Need'.",
    "The abstract introduces the Transformer, a new network architecture based solely on attention mechanisms that is more parallelizable and requires significantly less time to train than existing models."
  ],
  "document_url": "https://arxiv.org/pdf/1706.03762.pdf",
  "message": "New document processed and vectors stored in database."
}
```

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for features or improvements:

- Open an issue to discuss  
- Fork the repository  
- Create a new branch  
- Make your changes  
- Submit a pull request

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

