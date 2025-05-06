import streamlit as st 
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
import pandas as pd
import os

st.title('CHATBOT')


import streamlit as st


os.environ['LANGSMITH_API_KEY'] = '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'
llm = Ollama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")

embed_model = OllamaEmbeddings(
    model="llama3.2:1b",
    base_url='http://127.0.0.1:11434'
)

import pandas as pd

data1 = pd.read_csv("/Users/sbusisophakathi/Downloads/train_data.csv")
data1 = data1[0:70]
data1['content'] =  data1['answer']

text1 = " ".join(data1['content'].values)


import pickle

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text1)

vector_store = Chroma.from_texts(chunks, embed_model)

with open('chunks.pkl', 'wb') as file:
    pickle.dump(chunks, file)


retriever = vector_store.as_retriever()
chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

prompt = st.chat_input("Say something")
if prompt:

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain) 
    template = """Please answer me using strictly only the context I provided.Give me full answers not summaries and stop mentioning context in your response""" + prompt
 
    response = retrieval_chain.invoke({"input": template})
    st.markdown(response['answer'])


