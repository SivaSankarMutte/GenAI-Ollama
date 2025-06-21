from langchain.llms import Ollama
from langchain.chains import RetrievalQA

def create_qa_chain(vectorstore):
    llm = Ollama(model="llama3")
    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
