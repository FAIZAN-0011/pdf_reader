import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def extract_text_from_docx(file):
    text = ""

    doc = Document(file)

    for para in doc.paragraphs:
        if para.text:
            text += para.text + "\n"

    return text


def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return text_splitter.split_text(text)


def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(text_chunks, embeddings)

    vector_store.save_local("faiss_index")

    return vector_store


def get_conversation_chain():

    prompt_template = """
    Use the following pieces of context to answer the question at the end.

    If you don't know the answer, say you don't know.

    {context}

    Question: {question}

    Answer:
    """

    model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0.3
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = load_qa_chain(
        model,
        chain_type="stuff",
        prompt=prompt
    )

    return chain


def user_input(user_question):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = new_db.similarity_search(user_question)

    chain = get_conversation_chain()

    response = chain.run(
        input_documents=docs,
        question=user_question
    )

    return response


def main():

    st.header("Chat with PDF and DOCX files")
    st.title("Chat with your PDF and DOCX files using GROQ")

    uploaded_file = st.file_uploader(
        "Upload a PDF or DOCX file",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:

        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf([uploaded_file])

        elif uploaded_file.name.endswith(".docx"):
            text = extract_text_from_docx(uploaded_file)

        else:
            text = ""

        if text:

            text_chunks = get_text_chunks(text)

            get_vector_store(text_chunks)

            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            
            for message in st.session_state.messages:

                with st.chat_message(message["role"]):
                    st.write(message["content"])

            
            query = st.chat_input("Ask a question")

            if query:

                
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })

                with st.chat_message("user"):
                    st.write(query)

                
                response = user_input(query)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

                with st.chat_message("assistant"):
                    st.write(response)

        else:
            st.error("Unable to extract text from the uploaded file.")


if __name__ == "__main__":
    main()