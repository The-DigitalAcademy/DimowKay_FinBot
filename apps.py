import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain


# Initialize Ollama LLM and embedding model
llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url='http://127.0.0.1:11434')

# File and vector store paths
data_path = "train_data.csv"
vector_store_path = "vector_store"  # Same directory as script and data

# Load and prepare data
data = pd.read_csv(data_path).head(100)
data['content'] = data['answer']
text = " ".join(data['content'].values)

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

# Initialize or load vector store
if not os.path.exists(vector_store_path):
    vector_store = Chroma.from_texts(chunks, embed_model)
    vector_store.persist(vector_store_path)
else:
    vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embed_model)

# Create retriever and QA chain
retriever = vector_store.as_retriever()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit interface
st.title("Fin$mart chat")
user_input = st.chat_input("Ask me anything related to finance...")

if user_input:
    # Greeting filter
    if any(greet in user_input.lower() for greet in ["hi", "hello", "hey"]):
        st.markdown("Hello! How can I help you with your finance-related question today?")
    else:
        # Create strict instruction template for domain control
        template = (
            "You are a finance-only assistant. "
            "Only use the context provided. "
            "If the question is not related to finance, respond with: "
            "'I'm specialized in finance and can't help with that.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\n"
            "Answer:"
        )

        # Override LangChain's default if desired:
        # combine_docs_chain.prompt.template = template

        response = retrieval_chain.invoke({"input": user_input})
        st.markdown(response["answer"])
