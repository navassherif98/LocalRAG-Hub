# LocalRAG Hub

**LocalRAG Hub** is a local, private Retrieval-Augmented Generation (RAG) app leveraging [LangChain](https://github.com/langchain-ai/langchain) and [Ollama](https://ollama.com/) for both LLM and embedding models.  
It supports both a modern Streamlit web interface and a minimal standalone Python script for learning and experimentation.

---

## Features

- **Supports CSV and PDF ingestion**
- **Builds a ChromaDB vector store** for semantic search & retrieval
- **Local LLM + Embedding** via Ollama (all queries remain private)
- **Flexible UI** with Streamlit app (`app.py`)
- **Minimal, CLI-only learning script** (`main_rag.py`) for direct RAG workflow understanding
- **Automatic model checking**: Warns if required model is missing in Ollama

---

## Requirements

- Python 3.8+
- [Ollama](https://ollama.com/) (must be installed and running locally)
- All Python requirements in `requirements.txt` (see below)

---

## Installation

1. **Clone this repository:**
    ```
    git clone <YOUR_REPO_URL>
    cd <REPO_FOLDER>
    ```

2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
    *(Streamlit is included by default.)*

3. **Prepare Ollama models:**  
   Pull at least one LLM and one embedding model:
    ```
    ollama pull llama3.2
    ollama pull mxbai-embed-large
    ```
   *(You may use any compatible Ollama models—just pull them first!)*

4. **Add your data:**  
   Place a `CSV` or `PDF` file in the project folder.

---

## Usage

### Streamlit Web UI

Launch the app with: streamlit run app.py

- Upload your CSV or PDF and interact through a user-friendly browser interface.
- Select or custom-type Ollama models and retrieve context-rich answers from your data.

### Minimal Command-Line Demo

Use the basic RAG workflow via terminal: python main_rag.py

- Follow prompts to load a file, pick models, and ask questions.
- Great for learning, prototypes, or embedding in other workflows.

---

## Model Availability

If you select a model in the UI that is **not yet pulled** in Ollama, you’ll see a sidebar warning:
- Run `ollama pull <modelname>` in your terminal, then retry.

---

## Example Q&A Session (CLI)

Enter path to CSV or PDF: sample.csv
Ollama LLM model (e.g. llama3.2): llama3.2
Ollama emb. model (e.g. mxbai-embed-large): mxbai-embed-large
Docs to retrieve: 4

Ask a question (or type 'exit'): What is NLP?
Answer: Natural Language Processing (NLP) is ...

---

## Troubleshooting

- **Model not available:**  
  See sidebar warning or CLI error and run ollama pull <modelname>
- **Dependency errors:** 
  Run pip install -r requirements.txt
- **PDF errors:**  
  Make sure the file is not open in another program and is readable by PyPDF2.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

**Enjoy experimenting and learning with LocalRAG Hub!**
  

