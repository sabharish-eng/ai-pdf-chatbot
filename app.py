import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

st.set_page_config(
    page_title="My AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 My  AI friend ")
st.write("Upload a PDF and ask anything about it!")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Reading your PDF..."):
        loader = PyPDFLoader("temp.pdf")
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    template = ChatPromptTemplate.from_template("""
    Answer the question based on the context below.
    If you don't know the answer, just say "I don't know".

    Context: {context}
    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | template
        | llm
        | StrOutputParser()
    )

    st.success("✅ PDF loaded! Ask your questions below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input("Ask anything about your PDF..."):
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        st.chat_message("user").write(user_input)

        with st.spinner("Thinking..."):
            answer = chain.invoke(user_input)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        st.chat_message("assistant").write(answer)