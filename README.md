# Intelligent Document Q&A API

An open-source, high-performance API that allows you to find answers within your documents using powerful language models. Ingest PDFs, DOCX files, or emails, and get back precise answers to your questions.

This project is designed to be simple to set up and use, acting as a robust backend for any application that needs document-based question-answering capabilities.

---

## ✨ Features

- **Multi-Format Support**  
  Natively handles PDF, DOCX, and EML files.

- **Persistent Storage**  
  Uses Supabase with pgvector to store document embeddings. Process a document once and query it forever without reprocessing.

- **High-Quality Answers**  
  Leverages state-of-the-art language models from Together AI and Google for accurate embedding and answer generation.

- **Asynchronous & Fast**  
  Built with FastAPI for high-performance, non-blocking I/O.

- **Easy to Deploy**  
  Ready to be containerized with Docker or deployed to any modern cloud hosting service.

---

## 🚀 Getting Started

Follow these steps to get the API server running on your local machine.

### ✅ Prerequisites

- Python 3.8+
- A Supabase account with a project created
- API keys from Google AI Studio and Together AI

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
TOGETHER_API_KEY="your_together_ai_api_key_here"

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

