import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Custom embedding class for SentenceTransformer
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Replace with your PDF path
pdf_path = "Holy_Geeta.pdf"  # Verify this path
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    text = "".join(page.extract_text() for page in reader.pages)

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_text(text)

# Use custom embeddings
model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
vector_store = FAISS.from_texts(chunks, embedding=model)
vector_store.save_local("geeta_index")
print("E-book indexed successfully!")