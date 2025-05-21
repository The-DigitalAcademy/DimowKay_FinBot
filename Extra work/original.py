
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

def update_user_profile(user_id, new_name, new_surname):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET name=%s, surname=%s WHERE user_id=%s", (new_name, new_surname, user_id))
    conn.commit()
    cur.close()
    conn.close()


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

def delete_chat_session(history_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM chat_history WHERE history_id=%s", (history_id,))
    conn.commit()
    cur.close()
    conn.close()

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
    if st.button("Login"):
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
            new_name = st.text_input("Edit Name", value=st.session_state.name)
            new_surname = st.text_input("Edit Surname", value=st.session_state.surname)
            if st.button("Update Profile"):
                update_user_profile(st.session_state.user_id, new_name, new_surname)
                st.session_state.name, st.session_state.surname = new_name, new_surname
                st.success("Profile updated successfully!")

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
            st.markdown(f"### Chat from {qa['created_at']}")
            with st.chat_message("user"):
                st.markdown(qa["question"])
            with st.chat_message("assistant"):
                st.markdown(qa["answer"])
        else:
            st.info("No chat history found.")

    # Suggested questions
    suggested = get_suggested_questions()
    if suggested:
        st.markdown("### Suggested Questions:")
        cols = st.columns(min(len(suggested), 5))  # Ensures flexibility
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
    
   #Initialize ongoing chat session in memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    #Display chat history
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])

    #Handle new user input
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

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        # Save to in-memory chat
        st.session_state.chat_history.append({"question": user_q, "answer": bot_response})

        # Optionally persist to database
        save_message(st.session_state.user_id, user_q, bot_response)