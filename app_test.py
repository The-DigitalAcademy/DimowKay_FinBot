import streamlit as st
import ollama
import pandas as pd
from difflib import get_close_matches

# Load dataset
df = pd.read_csv("data/train_data.csv")

# Function to find the best matching question from the dataset
def find_best_match(user_query, questions, threshold=0.6):
    matches = get_close_matches(user_query, questions, n=1, cutoff=threshold)
    return matches[0] if matches else None

# Streamlit UI Setup
st.title("💰 FinBot - Your AI Financial Assistant")
st.write("Ask FinBot your financial questions!")

# User Input
user_input = st.text_input("Enter your financial question here:")

# Generate response using Ollama with dataset context
if st.button("Get Answer"):
    best_match = find_best_match(user_input, df["question"].tolist())

    if best_match:
        # Retrieve expected answer from the dataset
        expected_answer = df[df["question"] == best_match]["answer"].values[0]
        st.write("📚 **Context-Based Answer:**")
        st.write(expected_answer)

        # Also ask Ollama for a model-generated response
        response = ollama.chat(model="llama3", messages=[
            {"role": "user", "content": user_input}
        ])
        st.write("💡 **FinBot's AI Response:**")
        st.write(response['message'])
    else:
        # If no match is found, only use Ollama
        response = ollama.chat(model="llama3", messages=[
            {"role": "user", "content": user_input}
        ])
        st.write("💡 **FinBot's AI Response:**")
        st.write(response['message'])
