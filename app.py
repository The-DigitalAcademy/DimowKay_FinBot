# (keep all your imports as-is)
import streamlit as st
import psycopg2
import bcrypt
import uuid
import json
from datetime import datetime
import os
import pickle
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain

# --- LLM + ENV SETUP ---
load_dotenv()
llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="https://56aa-41-193-198-106.ngrok-free.app")
embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url="https://56aa-41-193-198-106.ngrok-free.app5")
# llm = Ollama(model="llama3.2:1b-instruct-q8_0", base_url="https://477b-41-193-198-106.ngrok-free.app ")
# embed_model = OllamaEmbeddings(model="llama3.2:1b-instruct-q8_0", base_url="https://477b-41-193-198-106.ngrok-free.app ")

vector_store_path = "vector_store.pkl"
if os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vector_store = pickle.load(f)
else:
    raise FileNotFoundError(f"{vector_store_path} not found.")

retriever = vector_store.as_retriever()
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# --- DB HELPERS (unchanged) ---
def get_connection():
    return psycopg2.connect(host="129.232.211.166", database="events", user="dylan", password="super123duper")

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS fin_users (
        user_id UUID PRIMARY KEY,
        name TEXT,
        surname TEXT,
        email TEXT UNIQUE,
        password TEXT,
        profile_pic TEXT
    );
    CREATE TABLE IF NOT EXISTS chat_history (
        history_id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES fin_users(user_id),
        qa_pair JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

def register_user(name, surname, email, password):
    user_id = str(uuid.uuid4())
    conn = get_connection()
    cur = conn.cursor()
    hashed_pw = hash_password(password).decode()
    try:
        cur.execute("INSERT INTO fin_users (user_id, name, surname, email, password) VALUES (%s, %s, %s, %s, %s)",
                    (user_id, name, surname, email, hashed_pw))
        conn.commit()
        return True
    except:
        return False
    finally:
        cur.close()
        conn.close()

def login_user(email, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, password, name, surname FROM fin_users WHERE email=%s", (email,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        user_id, hashed_pw, name, surname = result
        if check_password(password, hashed_pw):
            return user_id, name, surname
    return None, None, None

def update_user_profile(user_id, new_name, new_surname):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE fin_users SET name=%s, surname=%s WHERE user_id=%s", (new_name, new_surname, user_id))
    conn.commit()
    cur.close()
    conn.close()

def get_chat_sessions(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT history_id, created_at, qa_pair FROM chat_history WHERE user_id=%s ORDER BY created_at DESC", (user_id,))
    sessions = cur.fetchall()
    cur.close()
    conn.close()
    return sessions

def get_chat_by_history_id(history_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT qa_pair FROM chat_history WHERE history_id=%s", (history_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else []

def save_message(user_id, question, answer, history_id=None):
    conn = get_connection()
    cur = conn.cursor()
    qa_entry = {
        "question": question,
        "answer": answer,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

    if history_id:
        cur.execute("SELECT qa_pair FROM chat_history WHERE history_id=%s", (history_id,))
        result = cur.fetchone()
        if result:
            qa_data = result[0]
            if isinstance(qa_data, list):
                qa_data.append(qa_entry)
            else:
                qa_data = [qa_data, qa_entry]
            cur.execute("UPDATE chat_history SET qa_pair=%s WHERE history_id=%s", (json.dumps(qa_data), history_id))
    else:
        cur.execute("INSERT INTO chat_history (user_id, qa_pair) VALUES (%s, %s) RETURNING history_id", (user_id, json.dumps([qa_entry])))
        history_id = cur.fetchone()[0]

    conn.commit()
    cur.close()
    conn.close()
    return history_id

# --- STREAMLIT SETUP ---
st.set_page_config("FinBot", layout="wide")
create_tables()

# --- REMOVE CHAT HISTORY BORDERS ---
st.markdown("""
    <style>
    .stButton > button {
        border: none !important;
        background: none !important;
        box-shadow: none !important;
    }
    .stButton {
        margin-bottom: 4px;
    }
    </style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "login"
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None
if "clear_view" not in st.session_state:
    st.session_state.clear_view = False

# --- LOGIN PAGE ---
if st.session_state.page == "login":
    st.title("Login to FinBot")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user_id, name, surname = login_user(email, password)
        if user_id:
            st.session_state.page = "chat"
            st.session_state.user_id = user_id
            st.session_state.name = name
            st.session_state.surname = surname
            st.session_state.email = email
        else:
            st.error("Invalid credentials.")
    if st.button("Register"):
        st.session_state.page = "register"

# --- REGISTER PAGE ---
elif st.session_state.page == "register":
    st.title("Register")
    name = st.text_input("Name")
    surname = st.text_input("Surname")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")
    if st.button("Submit"):
        if password == confirm:
            success = register_user(name, surname, email, password)
            if success:
                st.success("Registered! Please log in.")
                st.session_state.page = "login"
            else:
                st.error("Email already exists.")
        else:
            st.error("Passwords don’t match.")
    if st.button("Login"):
        st.session_state.page = "login"

# --- CHAT PAGE ---
elif st.session_state.page == "chat":
    col1, col2 = st.columns([7, 2])
    with col1:
        logo_path = os.path.join("logo.jpg")
        st.image(logo_path, width=290)
        st.markdown("#### Your Personal Financial Advisor")
    with col2:
        if st.button(f"👤 {st.session_state.name}", key="profile_button"):
            st.session_state.page = "profile"
            st.rerun()

    st.sidebar.title("Chat History")
    if st.sidebar.button("New Chat"):
        st.session_state.clear_view = True
        st.session_state.selected_history_id = None

    sessions = get_chat_sessions(st.session_state.user_id)
    for history_id, ts, qa_list in sessions:
        try:
            first_question = qa_list[0]["question"]
        except (IndexError, KeyError, TypeError):
            first_question = "Untitled Chat"
        label = f"{first_question[:50]}..." if len(first_question) > 50 else first_question
        if st.sidebar.button(label, key=f"session_{history_id}"):
            st.session_state.selected_history_id = history_id
            st.session_state.clear_view = False

    if st.session_state.selected_history_id and not st.session_state.clear_view:
        chat_data = get_chat_by_history_id(st.session_state.selected_history_id)
        for msg in chat_data:
            with st.chat_message("user"):
                st.markdown(msg["question"])
            with st.chat_message("assistant"):
                st.markdown(msg["answer"])

    user_q = st.chat_input("Ask a finance question...")

    def validate_financial_answer(answer):
        validation_prompt = f"You are a finance-only AI. Is this answer related to finance?\nAnswer: {answer}\nReply Yes or No."
        result = llm.generate([validation_prompt])
        return "yes" in result.generations[0][0].text.lower()

    if user_q:
        with st.chat_message("user"):
            st.markdown(user_q)

        if user_q.lower().startswith(("hi", "hello", "hey")):
            bot_response = "Hello! How can I help you with your finance-related question today?"
        else:
            context = """ "You are a highly knowledgeable AI assistant specializing strictly in finance.\n"
        "Strictly answer only financially related topics.\n"
        "Never answer questions that are not financially related.\n"
        "Do not answer anything outside finance.\n"
        "Always provide accurate, objective, and concise answers to financial questions.\n"
        "Avoid unnecessary elaboration and focus directly on answering the user's query.\n"
        "Use the background context only if it is accurate, clear, and relevant. If the context is unclear, incomplete, low-quality, or irrelevant, ignore it and generate your own correct, concise financial answer.\n"
        "Do not copy or repeat the context verbatim — instead, synthesize your own response based on it.\n"
        "Do not speculate or use personal phrases like 'I think' or 'In my opinion'.\n"
        "If a user repeats a question, rephrase your response differently but keep the meaning and context consistent.\n"
        "Always answer only questions clearly related to finance — including personal finance, investing, credit, banking, insurance, budgeting, financial planning, macroeconomics, and markets.\n"
        "Provide answers that are well-explained, logically structured, and easy to understand.\n"
        "Write in a neutral and professional tone — avoid being casual or overly brief.\n"
        "If a retrieved answer is based on personal experience or opinion, do not display it — instead, generate your own accurate and objective financial response.\n"
        "If a valid financial question is asked, always answer — never refuse or say 'I can't help with that.'\n"
        "If a question is unrelated to finance, respond: 'I'm specialized in finance and can't help with that.'\n"
        "Do not provide medical, legal, or general life advice.\n"
        "Avoid incomplete explanations.\n"
        "Avoid unnecessary elaboration — focus on clarity and usefulness.\n"
        "Your answers must sound like a trustworthy, human financial advisor with deep domain knowledge.\n"
        "Structure complex answers with examples, bullet points, or breakdowns if needed.\n"
        "Do not answer questions that are vague, ambiguous, or lacking sufficient context.\n"
        "If a question is unclear, ask clarifying questions to get more information.\n"
        "Do not repeat the same answer for different questions — generate unique responses.\n"
        "Do not use overly technical jargon — explain concepts.\n"
        "Do not answer in general terms — stick to finance related terminologies.\n"
        "Do not interpret greetings, vague messages, or incomplete inputs as financial queries.\n"
        "Do not hallucinate questions or inject answers.\n"
        "Do not assume the user's intent.\n"
        "If the question is not financial related, respond: 'I'm specialized in finance and can't help with that. How can I assist you with a finance-related question today?'\n"
        "If the user uses a greeting such as 'Hi', 'Hello', or 'Hey', respond with: 'Hello! How can I help you with your finance-related question today?'\n"
        "You are a finance chatbot. Only respond to direct user questions. Do not create or simulate both sides of a conversation. Do not provide additional Q&A pairs unless asked.\n"
        """
            user_prompt = f"{context}\nUser Question: {user_q}"
            bot_response = retrieval_chain.invoke({"input": user_prompt})["answer"]

            if not validate_financial_answer(bot_response):
                bot_response = "I'm specialized in finance and can't help with that. Feel free to ask a finance-related question!"

        new_history_id = save_message(
            st.session_state.user_id,
            user_q,
            bot_response,
            history_id=st.session_state.selected_history_id if not st.session_state.clear_view else None
        )

        st.session_state.selected_history_id = new_history_id
        st.session_state.clear_view = False
        st.rerun()

    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.page = "login"
        st.rerun()

# --- PROFILE PAGE ---
elif st.session_state.page == "profile":
    st.sidebar.title("Navigation")
    if st.sidebar.button("← Back to Chat"):
        st.session_state.page = "chat"
        st.rerun()

    st.title("👤 Your Profile")
    st.text_input("Email", st.session_state.email, disabled=True)
    new_name = st.text_input("First Name", st.session_state.name)
    new_surname = st.text_input("Last Name", st.session_state.surname)

    if st.button("Save Changes"):
        update_user_profile(st.session_state.user_id, new_name, new_surname)
        st.session_state.name = new_name
        st.session_state.surname = new_surname
        st.success("Profile updated!")
