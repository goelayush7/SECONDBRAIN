# 🧠 Second Brain – RAG-Powered Knowledge Assistant

A personal **Second Brain** system that helps you store, search, and interact with your knowledge using **Retrieval-Augmented Generation (RAG)**. It connects a **cloud backend** for storage and indexing, a **Streamlit front-end** for UI, and intelligent **RAG pipelines** for answering questions using your uploaded data.

---

## 🚀 Features

- ✅ Upload documents (PDF, text, Markdown, etc.)
- 🔍 Semantic search over your knowledge base
- 💬 Ask questions and get context-aware responses
- ☁️ Cloud-based storage & retrieval
- ⚡ RAG-powered responses using OpenAI or Hugging Face models
- 🎛️ Streamlit frontend with interactive UI

---

## 🧱 Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **Backend / RAG Layer:** [LangChain](https://www.langchain.com/), [FAISS](https://github.com/facebookresearch/faiss) or [ChromaDB](https://www.trychroma.com/)
- **LLM APIs:** GROQ / Hugging Face Transformers
- **Cloud:** AWS
- **Storage:** Cloud Bucket (e.g., S3), or local DB if self-hosted

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/second-brain-rag.git
cd second-brain-rag

### 2.Download Requirments.txt

### 3. Add ENV VARS

