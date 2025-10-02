import pandas as pd
import PyPDF2
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DB_DIR = "./chroma_langchain_db"

def load_csv_or_pdf(filepath):
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".pdf"):
        reader = PyPDF2.PdfReader(open(filepath, "rb"))
        chunks = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                chunks.append({"text": text, "page": i+1})
        return pd.DataFrame(chunks)
    else:
        raise ValueError("Unsupported file. Please use .csv or .pdf")

def vectorize_docs(df, embedding_model):
    if "text" not in df.columns:
        # Assume first column is the main text column if not labeled 'text'
        df = df.rename(columns={df.columns[0]: "text"})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.create_documents(df["text"].astype(str).tolist())
    return docs

def create_chroma_db(docs, embedding_model_name, persist_dir):
    embedder = OllamaEmbeddings(model=embedding_model_name)
    vectordb = Chroma.from_documents(docs, embedding=embedder, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

def get_top_k(dbase, question, embedding_model, k=4):
    embedder = OllamaEmbeddings(model=embedding_model)
    retriever = dbase.as_retriever(search_kwargs={'k': k}, embedding=embedder)
    relevant_docs = retriever.invoke(question)
    return "\n\n---\n\n".join(d.page_content for d in relevant_docs)

def main():
    filepath = input("Enter path to CSV or PDF: ").strip()
    model_name = input("Ollama LLM model (e.g. llama3.2): ").strip()
    embedding_model = input("Ollama emb. model (e.g. mxbai-embed-large): ").strip()
    top_k = int(input("Docs to retrieve: ").strip() or 4)

    print("Loading and preparing data...")
    df = load_csv_or_pdf(filepath)
    docs = vectorize_docs(df, embedding_model)
    vectordb = create_chroma_db(docs, embedding_model, DB_DIR)

    llm = OllamaLLM(model=model_name)
    prompt = ChatPromptTemplate.from_template("""
You are an expert in answering questions about our uploaded documents.

Here are some relevant docs or reviews: {reviews}

Here is the question to answer: {question}
""")

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if not question or question.lower() == "exit":
            break
        context = get_top_k(vectordb, question, embedding_model, k=top_k)
        result = prompt | llm
        answer = result.invoke({"reviews": context, "question": question})
        print("\nAnswer:", answer, "\n" + "-" * 40)

if __name__ == "__main__":
    main()
