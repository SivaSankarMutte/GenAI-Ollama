from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings

def create_vectorstore(documents):
    embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(documents, embeddings)
    return db
