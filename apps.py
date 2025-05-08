import streamlit as st
import os
import pandas as pd
import pickle

from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

# Paths
VECTORSTORE_DIR = "vectorstore"
PICKLE_FILE = "retriever.pkl"
DATA_FILE = "train_data.csv"

# Set environment variable
os.environ['LANGSMITH_API_KEY'] = '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'

# Load models
llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")
embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url="http://127.0.0.1:11434")

# Load or create vector store and retriever
if os.path.exists(PICKLE_FILE):
    with open(PICKLE_FILE, "rb") as f:
        retriever = pickle.load(f)
else:
    data = pd.read_csv(DATA_FILE).head(100)
    data['content'] = data['answer']
    text = " ".join(data['content'].values)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    chunks = text_splitter.split_text(text)

    vector_store = Chroma.from_texts(chunks, embed_model, persist_directory=VECTORSTORE_DIR)
    vector_store.persist()

    retriever = vector_store.as_retriever()

    with open(PICKLE_FILE, "wb") as f:
        pickle.dump(retriever, f)

# Load chain template
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

# In-memory recent query tracker
recent_questions = {}

# LLM-based classifier
def is_finance_related(query):
    classification_prompt = (
        f"Is the following question financially related? "
        f"Just answer YES or NO. Question: {query}"
    )
    response = llm.invoke(classification_prompt).strip().upper()
    return "YES" in response

# Prompt builder
def build_finance_prompt(context, question, repeat_count):
    temperature = min(0.7 + 0.1 * (repeat_count - 1), 1.0)
    instruction = (
        "You are a highly knowledgeable AI assistant specializing strictly in finance.\n"
        "Strictly answer only financially related topics.\n"
        "Never answer questions that are not financially related.\n"
        "Do not answer anything outside finance.\n"
        "Always provide accurate, objective, and concise answers to financial questions.\n"
        "Avoid unnecessary elaboration and focus directly on answering the user's query.\n"
        "Use the following related financial questions and answers as background knowledge to inform your response.\n"
        "Do not copy or repeat the context verbatim — instead, synthesize your own response based on it.\n"
        "Do not speculate or use personal phrases like 'I think' or 'In my opinion'.\n"
        "If a valid financial question is asked, always answer — never refuse or say 'I can't help with that.'\n"
        "If a question is unrelated to finance, respond: 'I'm specialized in finance and can't help with that.'\n"
    )

    return f"""{instruction}

Background context:
{context}

User question: {question}

Answer:""", temperature

# Streamlit UI
st.title("💰 Finance QA Bot")
prompt = st.chat_input("Ask a finance-related question")

if prompt:
    normalized_query = prompt.lower().strip()
    count = recent_questions.get(normalized_query, 0) + 1
    recent_questions[normalized_query] = count

    if not is_finance_related(prompt):
        st.markdown("I'm specialized in finance and can't help with that. Please ask a finance-related question.")
    else:
        relevant_docs = retriever.get_relevant_documents(prompt)
        context = "\n".join([doc.page_content for doc in relevant_docs])

        full_prompt, temperature = build_finance_prompt(context, prompt, count)
        response = llm.invoke(full_prompt)
        st.markdown(response)
