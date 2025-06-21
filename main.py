from app.loader import load_and_split_pdf
from app.vectorstore import create_vectorstore
from app.qa_chain_local import create_qa_chain

def main():
    docs = load_and_split_pdf("document.pdf")
    vs = create_vectorstore(docs)
    qa = create_qa_chain(vs)

    while True:
        question = input("\nAsk something (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        answer = qa.run(question)
        print("\nAnswer:", answer)

if __name__ == "__main__":
    main()
