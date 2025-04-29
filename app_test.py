import streamlit as st
import ollama
import pandas as pd
from difflib import get_close_matches


df = pd.read_csv("data/train_data.csv")


def find_best_match(user_query, questions, threshold=0.6):
    matches = get_close_matches(user_query, questions, n=1, cutoff=threshold)
    return matches[0] if matches else None


st.title("FinBot - Your AI Financial Assistant")
st.write("Ask FinBot your financial questions!")


user_input = st.text_input("Enter your financial question here:")


if st.button("Get Answer"):
    best_match = find_best_match(user_input, df["question"].tolist())

    if best_match:

        expected_answer = df[df["question"] == best_match]["answer"].values[0]
        st.write("Expected Answer")
        st.write(expected_answer)

        response = ollama.chat(model="llama3", messages=[
            {"role": "user", "content": user_input}
        ])
        st.write("Model's Response")
        st.write(response['message'])
    else:
        response = ollama.chat(model="llama3", messages=[
            {"role": "user", "content": user_input}
        ])
        st.write("💡 **FinBot's AI Response:**")
        st.write(response['message'])
