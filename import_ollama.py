import ollama
import pandas as pd

# Load your CSV
df = pd.read_csv('/Users/tshmacm1172/Desktop/DimowKay_FinBot/data/train_data.csv')

# Build few-shot prompt
shots = ""
for _, row in df.head(3).iterrows():  # Use top 3 examples as few-shots
    shots += f"Q: {row['question']}\nA: {row['answer']}\n\n"

# New question to test
user_question = "What is GDP?"

# Full prompt
full_prompt = shots + f"Q: {user_question}\nA:"

# Use Ollama (make sure the app is running)
response = ollama.chat(
    model='llama3',
    messages=[
        {"role": "user", "content": full_prompt}
    ]
)

print("Answer:", response['message']['content'])
