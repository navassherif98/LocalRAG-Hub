# vector.py

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

def get_retriever(
        df,
        embedding_model="mxbai-embed-large",
        collection_name="uploaded_docs",
        num_docs=5,
        db_location="./chroma_langchain_db"
    ):
    """
    Create a ChromaDB retriever from a Pandas DataFrame.
    df: DataFrame with columns ['Title', 'Review', 'Rating', 'Date'] or ['text']
    embedding_model: Ollama embedding model string
    collection_name: ChromaDB collection
    num_docs: number of docs to retrieve per query
    db_location: persistent chroma database directory
    Returns: retriever object
    """
    embeddings = OllamaEmbeddings(model=embedding_model)

    # If initiating new ChromaDB collection, add documents
    add_documents = not db_location or not os.path.exists(db_location)
    documents = []
    ids = []

    # Accept diverse file formats (CSV with review columns or text column)
    for i, row in df.iterrows():
        text = (
            row.get("Title", "") + " " +
            row.get("Review", "") if "Review" in row else
            str(row.get("text", row))
        )
        metadata = {}
        for col in ["Rating", "Date"]:
            if col in row:
                metadata[col] = row[col]
        doc = Document(
            page_content=str(text),
            metadata=metadata,
            id=str(i)
        )
        ids.append(str(i))
        documents.append(doc)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings
    )

    # Only add docs when database is new/empty
    if add_documents:
        vector_store.add_documents(documents=documents, ids=ids)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": num_docs}
    )

    return retriever
