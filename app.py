import streamlit as st
import pandas as pd
from vector import get_retriever
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import PyPDF2
import subprocess

def is_ollama_model_available(model_name):
    try:
        proc = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return model_name in proc.stdout
    except Exception:
        return False

# ===== Sidebar Top: Application Name and File Upload =====
st.sidebar.title("LocalRAG Hub")
app_name = st.sidebar.text_input("Application Name", value="LocalRAG Hub")
uploaded_file = st.sidebar.file_uploader("Upload CSV or PDF", type=["csv", "pdf"])

# ===== Model Selection with Dropdown + Custom Option =====
llm_options = ["llama3.2", "llama2", "phi3", "mistral", "Custom"]
embedding_options = ["mxbai-embed-large", "mxbai-embed-small", "Custom"]

model_choice = st.sidebar.selectbox("LLM Model Name (Ollama)", llm_options)
if model_choice == "Custom":
    model_name = st.sidebar.text_input("Enter custom LLM model name")
else:
    model_name = model_choice

embedding_choice = st.sidebar.selectbox("Embedding Model", embedding_options)
if embedding_choice == "Custom":
    embedding_model = st.sidebar.text_input("Enter custom embedding model")
else:
    embedding_model = embedding_choice

# --- Check LLM availability
if model_name and model_name != "Custom" and not is_ollama_model_available(model_name):
    st.sidebar.warning(
        f"Model '{model_name}' is not found locally!\n"
        f"Please run:\n\n  ollama pull {model_name}\n\nin your terminal before using this model."
    )

# --- Check Embedding model availability
if embedding_model and embedding_model != "Custom" and not is_ollama_model_available(embedding_model):
    st.sidebar.warning(
        f"Embedding model '{embedding_model}' is not found locally!\n"
        f"Please run:\n\n  ollama pull {embedding_model}\n\nin your terminal before using this embedding model."
    )

num_docs = st.sidebar.slider("Docs to retrieve per question", 1, 10, 5)

col1, col2 = st.columns([1, 12])
with col1:
    st.image("assets/robot.png", width=80)
with col2:
    st.markdown(f"# {app_name}")

# ===== PDF Extraction Helper =====
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    page_chunks = [{"text": page.extract_text(), "page": i+1}
                   for i, page in enumerate(reader.pages) if page.extract_text()]
    return pd.DataFrame(page_chunks)

retriever = None
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.pdf'):
            df = extract_text_from_pdf(uploaded_file)
            st.success(f"Loaded PDF with {len(df)} page chunks!")
        else:
            df = pd.read_csv(uploaded_file)
            st.success(f"Loaded CSV with {len(df)} records!")
        retriever = get_retriever(
            df,
            embedding_model=embedding_model,
            num_docs=num_docs,
            collection_name="user_data"
        )
    except Exception as e:
        st.error(f"Could not load dataset: {str(e)}")

model = OllamaLLM(model=model_name)
template = """
You are an expert in answering questions about our uploaded documents.

Here are some relevant docs or reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ===== Chat Input Form =====
if retriever:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("Ask a question", key="question")
        submitted = st.form_submit_button("Submit")
        if submitted and question.strip():
            with st.spinner("Processing, please wait..."):
                reviews = retriever.invoke(question)
                result = chain.invoke({"reviews": reviews, "question": question})
                st.session_state.chat_history.append((question, result))


    # === Display Q&A History (latest at top, rest below divider) ===
    if st.session_state.chat_history:
        latest_q, latest_a = st.session_state.chat_history[-1]
        st.markdown(f"<span style='font-size:1.4em;font-weight:bold;'>Q: {latest_q}</span>", unsafe_allow_html=True)
        st.markdown(f"**A:** {latest_a}")

        if len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown("### Previous Q&A")
            for q, a in reversed(st.session_state.chat_history[:-1]):
                st.markdown(f"<span style='font-size:1.2em;font-weight:bold;'>Q: {q}</span>", unsafe_allow_html=True)
                st.markdown(f"**A:** {a}")
    else:
        st.info("No Q&A yet—ask your first question above!")

else:
    st.info("Please upload a CSV or PDF dataset to get started.")

st.sidebar.markdown("""
Created with LocalRAG Hub.
- Use dropdowns or choose "Custom" to type your own model/embedding name as available in Ollama.
- To add a new model: Run `ollama pull <modelname>` in your terminal.
""")
