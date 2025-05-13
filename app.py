
import streamlit as st
import psycopg2
import bcrypt
import uuid
import json
from datetime import datetime

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


# --- DB CONNECTION ---
def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="finbot",
        user="postgres",
        password="none"
    )

# --- AUTH HELPERS ---
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

# --- DB SETUP ---
def create_tables():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id UUID PRIMARY KEY,
        name TEXT,
        surname TEXT,
        email TEXT UNIQUE,
        password TEXT,
        profile_pic TEXT
    );

    CREATE TABLE IF NOT EXISTS chat_history (
        history_id SERIAL PRIMARY KEY,
        user_id UUID REFERENCES users(user_id),
        qa_pair JSONB NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS suggested_questions (
        id SERIAL PRIMARY KEY,
        question TEXT UNIQUE NOT NULL,
        answer TEXT NOT NULL
    );
    """)
    conn.commit()
    cur.close()
    conn.close()

# --- AUTH FUNCTIONS ---
def register_user(name, surname, email, password):
    user_id = str(uuid.uuid4())
    conn = get_connection()
    cur = conn.cursor()
    hashed_pw = hash_password(password).decode()
    try:
        cur.execute("INSERT INTO users (user_id, name, surname, email, password) VALUES (%s, %s, %s, %s, %s)",
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
    cur.execute("SELECT user_id, password, name, surname FROM users WHERE email=%s", (email,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    if result:
        user_id, hashed_pw, name, surname = result
        if check_password(password, hashed_pw):
            return user_id, name, surname
    return None, None, None

def get_chat_sessions(user_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT history_id, created_at, qa_pair
        FROM chat_history
        WHERE user_id=%s
        ORDER BY created_at DESC
    """, (user_id,))
    sessions = cur.fetchall()
    cur.close()
    conn.close()
    return sessions

def get_chat_by_history_id(history_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT qa_pair FROM chat_history
        WHERE history_id=%s
    """, (history_id,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result[0] if result else None

def get_suggested_questions():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT question, answer FROM suggested_questions LIMIT 5")
    questions = cur.fetchall()
    cur.close()
    conn.close()
    return questions

def save_message(user_id, question, answer):
    qa = json.dumps({"question": question, "answer": answer, "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO chat_history (user_id, qa_pair) VALUES (%s, %s)", (user_id, qa))
    conn.commit()
    cur.close()
    conn.close()

# --- INIT ---
st.set_page_config("FinBot", layout="wide")
create_tables()

if "page" not in st.session_state:
    st.session_state.page = "login"
if "show_profile" not in st.session_state:
    st.session_state.show_profile = False
if "clear_view" not in st.session_state:
    st.session_state.clear_view = False
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None

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
            st.session_state.clear_view = False
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
                st.success("Registered. Please login.")
                st.session_state.page = "login"
            else:
                st.error("Email already exists.")
        else:
            st.error("Passwords don't match.")
    if st.button("Back to Login"):
        st.session_state.page = "login"

# --- CHAT PAGE ---
elif st.session_state.page == "chat":
    # Top layout
    col1, col2 = st.columns([7, 2])
    with col1:
        logo_path = "/Users/tshmacm1171/Desktop/DimowKay_FinBot/logo.jpg"
        logo = st.image(logo_path, width=290)
        st.markdown("#### Your Personal Financial Advisor")
    with col2:
        if st.button(f"👤 {st.session_state.name}", key="profile_button"):
            st.session_state.show_profile = not st.session_state.show_profile
        if st.session_state.show_profile:
            st.markdown(f"**Name:** {st.session_state.name} {st.session_state.surname}")
            st.markdown(f"**Email:** {st.session_state.email}")

    # Sidebar with fixed layout
    with st.sidebar:
       
        st.markdown("<div style='position:sticky; top:0; background-color:white; z-index:10;'>", unsafe_allow_html=True)
        if st.button("New Chat"):
            st.session_state.clear_view = True
            st.success("Chat cleared from view.")
        st.markdown("</div>", unsafe_allow_html=True)

        # Scrollable chat history
        st.markdown("<h2>Chat History</h2>", unsafe_allow_html=True)
        chat_history_container = st.container()
        with chat_history_container:
            sessions = get_chat_sessions(st.session_state.user_id)
            if sessions:
                for history_id, ts, qa_json in sessions:
                    qa = qa_json
                    timestamp = ts.strftime("%Y-%m-%d %H:%M")
                    question = qa.get("question", "")
                    label = f"{timestamp}\n{question}"
                    if st.button(label, key=f"session_{history_id}"):
                        st.session_state.selected_history_id = history_id
                        st.session_state.clear_view = False

        # Fixed "Logout" button at the bottom
        st.markdown("<div style='position:sticky; bottom:0; background-color:white; z-index:10;'>", unsafe_allow_html=True)
        if st.button("Logout"):
            st.session_state.clear()
            st.session_state.page = "login"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # Main chat area
    if st.session_state.selected_history_id and not st.session_state.clear_view:
        qa = get_chat_by_history_id(st.session_state.selected_history_id)
        if qa:
            st.markdown(f"###  Chat from {qa['created_at']}")
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                st.markdown(qa["answer"])
        else:
            st.info("No chat history found.")

    # Suggested questions
    suggested = get_suggested_questions()
    if suggested:
        st.markdown("###  Suggested Questions:")
        cols = st.columns(5)
        for i, (q, a) in enumerate(suggested):
            if cols[i].button(q):
                save_message(st.session_state.user_id, q, a)
                st.rerun()

    # Input box for user questions
    user_q = st.chat_input("Ask me anything related to finance...")

    def validate_financial_answer(answer):
        validation_prompt = (
            "You are a highly knowledgeable AI assistant specializing strictly in finance.\n"
            "Is the following answer financially related?\n"
            "Answer: " + answer + "\n"
            "Only answer with 'Yes' or 'No'."
        )
        check_response = llm.generate([validation_prompt])
        return "yes" in check_response.generations[0][0].text.lower()
    
    if user_q:
        if any(greet in user_q.lower() for greet in ["hi", "hello", "hey"]):
            st.markdown("Hello! How can I help you with your finance-related question today?")
        else:
            bot_a = retrieval_chain.invoke({"input": user_q})["answer"]
            if validate_financial_answer(bot_a):
                st.markdown(bot_a)
            else:
                st.markdown("I'm specialized in finance and can't help with that. How can I assist you with a finance-related question today?")
            save_message(st.session_state.user_id, user_q, bot_a)
            st.rerun()