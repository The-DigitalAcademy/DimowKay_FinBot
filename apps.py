import streamlit as st
import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize Ollama model for embeddings and LLM
embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")
llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")

# Load the financial Q&A data
data_path = "/Users/tshmacm1172/Desktop/DimowKay_FinBot/data/train_data.csv"
data = pd.read_csv(data_path).head(100)  # Use top 100 rows for demo

# Combine questions and answers into a single text column
data['content'] = data['answer']
text = " ".join(data['content'].values)

# Split the data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

# Generate embeddings for each chunk
embeddings = embed_model.embed_documents(chunks)

# Initialize Annoy Index (dense vector search)
dimension = len(embeddings[0])  # Assuming all embeddings are the same length
index = AnnoyIndex(dimension, 'angular')  # You can choose other distances like 'euclidean', 'manhattan', etc.

# Add vectors to the index
for i, embedding in enumerate(embeddings):
    index.add_item(i, embedding)

# Build the index
index.build(10)  # You can set the number of trees (higher value = better accuracy)

# Save the Annoy index to disk
index.save("annoy_index.ann")

# Load the Annoy index from disk
index_loaded = AnnoyIndex(dimension, 'angular')
index_loaded.load("annoy_index.ann")

# Define the search function
def search_annoy(query, top_k=3):
    query_embedding = embed_model.embed_documents([query])[0]
    return index_loaded.get_nns_by_vector(query_embedding, top_k)

# Setup the conversation retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, search_annoy)

# Function to interact with the bot
def ask_finance_bot(user_query):
    response = qa_chain.run(input=user_query)
    return response

# Streamlit App Interface
st.title("Financial QA Chatbot")
st.write("Ask any financial question, and I will assist you based on the knowledge I have.")

# User input for question
user_query = st.text_input("Enter your financial question:")

if user_query:
    answer = ask_finance_bot(user_query)
    st.write(f"**Answer:** {answer}")

# Optional: Allow user to see the chunks being used for context (for debugging)
if st.checkbox('Show context used for answer'):
    st.write("### Context Used:")
    st.write(chunks[:3])  # Show the first 3 chunks for reference
