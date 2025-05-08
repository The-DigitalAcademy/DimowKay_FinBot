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

# Load Data
data_path = "train_data.csv"
data1 = pd.read_csv(data_path)
data1 = data1[0:100]
data1['content'] = data1['answer']

text1 = " ".join(data1['content'].values)

# Split Text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text1)

# Vector Store
vector_store = Chroma.from_texts(chunks, embed_model)
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
