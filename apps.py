import streamlit as st
import os
import pandas as pd
import pickle
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

# Set API Key (if applicable)
load_dotenv()

# Initialize Ollama LLM and embedding model
llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url='http://127.0.0.1:11434')

# Load pre-vectorized data from `vector.pkl`
vector_store_path = "/Users/tshmacm1171/Desktop/DimowKay_FinBot/vector_store.pkl"

if os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vector_store = pickle.load(f)
    print("Loaded pre-vectorized data successfully.")
else:
    raise FileNotFoundError(f" Vector file {vector_store_path} not found! Ensure you have run the vectorizing step and saved it.")

# Create retriever
retriever = vector_store.as_retriever()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit Interface
st.title("Fin$mart Chat")
user_input = st.chat_input("Ask me anything related to finance...")

def validate_financial_answer(answer):
    validation_prompt = (
        "You are a highly knowledgeable AI assistant specializing strictly in finance.\n"
        "Is the following answer financially related?\n"
        "Answer: " + answer + "\n"
        "Only answer with 'Yes' or 'No'."
    )
    check_response = llm.generate([validation_prompt])
    return "yes" in check_response.generations[0][0].text.lower()

if user_input:
    if any(greet in user_input.lower() for greet in ["hi", "hello", "hey"]):
        st.markdown("Hello! How can I help you with your finance-related question today?")
    else:
        response = retrieval_chain.invoke({"input": user_input})["answer"]
        if validate_financial_answer(response):
            st.markdown(response)
        else:
            st.markdown("I'm specialized in finance and can't help with that. How can I assist you with a finance-related question today?")
