import streamlit as st
import os
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Custom embedding class for SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Check if geeta_index exists, else generate it
if not os.path.exists("geeta_index"):
    pdf_path = "Holy_Geeta.pdf"  # PDF must be in the repository root
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages)
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_text(text)
        model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(chunks, embedding=model)
        vector_store.save_local("geeta_index")
    except FileNotFoundError:
        st.error("Holy_Geeta.pdf not found in the repository. Please upload it.")
        st.stop()

# Initialize LLM with OpenAI API key from secrets
llm = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load vector store
model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
vector_store = FAISS.load_local("geeta_index", embeddings=model, allow_dangerous_deserialization=True)

# Define PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are a chatbot that answers questions based only on *The Holy Geeta* by Swami Chinmayananda. 
    Use only the provided context: {context}. 
    Answer the question: {question} in a concise manner, citing specific chapters and verses (e.g., Chapter 2, Verse 47) when possible. 
    If the answer is not in the context, say, “This is not covered in *The Holy Geeta*.” 
    Do not use external knowledge.
    """
)

# Define qa_chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)

# Streamlit UI with chat history
st.set_page_config(page_title="Holy Geeta Chatbot", page_icon="📖")
st.title("Holy Geeta Chatbot")
st.markdown("Ask questions about *The Holy Geeta* by Swami Chinmayananda")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
query = st.chat_input("Ask a question about The Holy Geeta:")
if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Get and display response
    with st.spinner("Fetching answer..."):
        response = qa_chain({"query": query})
    with st.chat_message("assistant"):
        st.markdown(response["result"])
    st.session_state.messages.append({"role": "assistant", "content": response["result"]})