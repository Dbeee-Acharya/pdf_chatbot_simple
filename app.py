import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import streamlit as st
import tempfile

# Load environment variables
load_dotenv()


class PDFQuestionAnswering:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.vector_store = None
        self.qa_chain = None

    def process_pdf(self, pdf_file):
        """Process PDF file and create vector store"""
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )
        texts = text_splitter.split_documents(pages)

        # Create vector store
        self.vector_store = FAISS.from_documents(texts, self.embeddings)

        # Initialize QA chain
        llm = OpenAI(temperature=0, openai_api_key=self.openai_api_key)
        self.qa_chain = load_qa_chain(llm, chain_type="stuff")

        os.unlink(tmp_path)

        return "PDF processed successfully!"

    def answer_question(self, question):
        """Answer question based on the processed PDF"""
        if not self.vector_store or not self.qa_chain:
            return "Please upload a PDF first!"

        # Search for relevant documents
        docs = self.vector_store.similarity_search(question)

        # Generate answer
        response = self.qa_chain.run(input_documents=docs, question=question)

        return response


# Streamlit UI
def main():
    st.title("PDF Question Answering System")

    # Initialize session state
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = PDFQuestionAnswering()

    # File upload
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                result = st.session_state.qa_system.process_pdf(uploaded_file)
                st.success(result)

    # Question answering
    st.header("2. Ask Questions")
    question = st.text_input("Enter your question:")

    if question:
        if st.button("Get Answer"):
            with st.spinner("Generating answer..."):
                answer = st.session_state.qa_system.answer_question(question)
                st.write("Answer:", answer)


if __name__ == "__main__":
    main()
