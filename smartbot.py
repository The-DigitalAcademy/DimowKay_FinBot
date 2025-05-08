import streamlit as st
import faiss
import numpy as np
import pandas as pd
import torch
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# --- HuggingFace login ---
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

# --- Load data ---
train = pd.read_csv('train_data.csv')  # Assumes it's in the same directory
questions = train['question'].tolist()
answers = train['answer'].tolist()

qa_pairs = [f"Q: {q} A: {a}" for q, a in zip(questions, answers)]

# --- Embedding model ---
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
answer_embeddings = embedding_model.encode(answers)

# --- FAISS index ---
index = faiss.IndexFlatL2(answer_embeddings.shape[1])
index.add(np.array(answer_embeddings))

# --- LLaMA model setup ---
model_name = "meta-llama/Llama-3-8B-Instruct"  # Update to a valid space-available model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# --- Helper Functions ---
def sanitize_answer(question, answer):
    return any(word.lower() in answer.lower() for word in question.lower().split())

recent_questions = {}

def is_finance_question(user_query):
    check_prompt = (
        f"You are a financial expert. Determine whether the following question is clearly about finance:\n\n"
        f"Question: {user_query}\n\n"
        f"Respond only with 'Yes' or 'No'."
    )
    input_ids = tokenizer(check_prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **input_ids,
        max_new_tokens=10,
        temperature=0.0,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return response.lower().startswith("yes")

def ask_finance_bot(user_query, top_k=3):
    normalized_query = user_query.lower().strip()
    count = recent_questions.get(normalized_query, 0) + 1
    recent_questions[normalized_query] = count

    # Embed user query
    query_embedding = embedding_model.encode([user_query])
    D, I = index.search(np.array(query_embedding), top_k)
    retrieved_answers = [answers[i] for i in I[0]]
    context = "\n".join([f"- {text}" for text in retrieved_answers])

    temperature = min(0.7 + 0.1 * (count - 1), 1.0)

    instruction = (
        "You are a highly knowledgeable AI assistant specializing strictly in finance.\n"
        "Strictly answer only financially related topics.\n"
        "Never answer questions that are not financially related.\n"
        "Do not answer anything outside finance.\n"
        "Always provide accurate, objective, and concise answers to financial questions.\n"
        "Avoid unnecessary elaboration and focus directly on answering the user's query.\n"
        "Use the background context only if it is accurate, clear, and relevant. If the context is unclear, incomplete, low-quality, or irrelevant, ignore it and generate your own correct, concise financial answer.\n"
        "Do not copy or repeat the context verbatim — instead, synthesize your own response based on it.\n"
        "Do not speculate or use personal phrases like 'I think' or 'In my opinion'.\n"
        "If a valid financial question is asked, always answer — never refuse or say 'I can't help with that.'\n"
        "If a question is unrelated to finance, respond: 'I'm specialized in finance and can't help with that. How can I assist you with a finance-related question today?'\n"
        "If a greeting like 'Hi', 'Hello', or 'Hey' is used, respond with: 'Hello! How can I help you with your finance-related question today?'\n"
    )

    for _ in range(6):
        prompt = f"""{instruction}

Background context:
{context}

User question: {user_query}

Answer:"""

        input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **input_ids,
            max_new_tokens=256,
            temperature=temperature,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer_text = response.split("Answer:")[-1].strip()

        if sanitize_answer(user_query, answer_text):
            return answer_text

    return "I'm not confident in the response. Please consult a certified financial expert."

# --- Streamlit App UI ---
st.set_page_config(page_title="DiMowkayBot - Finance Assistant", layout="centered")
st.title("💸 DiMowkayBot - Your Finance Q&A Assistant")

user_query = st.text_input("Enter your finance-related question:")

if user_query:
    if not is_finance_question(user_query):
        st.warning("I'm specialized in finance and can't help with that. How can I assist you with a finance-related question today?")
    else:
        with st.spinner("Thinking..."):
            answer = ask_finance_bot(user_query)
        st.success("Response:")
        st.write(answer)
