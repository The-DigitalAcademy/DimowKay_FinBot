
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer


train_data = pd.read_csv('/workspaces/DimowKay_FinBot/train_data.csv')
val_data = pd.read_csv('/workspaces/DimowKay_FinBot/test_data.csv')
test_data = pd.read_csv('/workspaces/DimowKay_FinBot/val_data.csv')


train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_data['question'], train_data['answer'], test_size=0.2, random_state=42
)


all_labels = pd.concat([train_labels, val_labels])


label_encoder = LabelEncoder()
label_encoder.fit(all_labels)


train_labels_encoded = label_encoder.transform(train_labels)
val_labels_encoded = label_encoder.transform(val_labels)


bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_encodings_bert = bert_tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings_bert = bert_tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)


gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
train_encodings_gpt = gpt_tokenizer(list(train_texts), truncation=True, padding=True, max_length=512)
val_encodings_gpt = gpt_tokenizer(list(val_texts), truncation=True, padding=True, max_length=512)

# Prepare Data for PyTorch:

import torch

# Dataset for BERT (classification tasks)
class FinancialDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Create datasets for BERT
train_dataset_bert = FinancialDataset(train_encodings_bert, train_labels_encoded)
val_dataset_bert = FinancialDataset(val_encodings_bert, val_labels_encoded)

# Dataset for GPT (generative tasks)
class FinancialDatasetGPT(torch.utils.data.Dataset):
    def __init__(self, tokenizer, texts, labels, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Tokenize input text
        inputs = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Tokenize labels
        labels = self.tokenizer(
            self.labels[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Convert to tensors and return
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        item['labels'] = labels['input_ids'].squeeze(0)
        return item

# Create datasets for GPT
train_dataset_gpt = FinancialDatasetGPT(
    tokenizer=gpt_tokenizer,
    texts=list(train_texts),
    labels=list(train_labels),
    max_length=512,
)

val_dataset_gpt = FinancialDatasetGPT(
    tokenizer=gpt_tokenizer,
    texts=list(val_texts),
    labels=list(val_labels),
    max_length=512,
)


 # Fine-Tune BERT:
#We will fine-tune a pretrained BERT model for classification tasks.

from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load BERT model for classification
bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_encoder.classes_))

# Define training arguments
training_args_bert = TrainingArguments(
    output_dir='./results_bert',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
)

# Fine-tune BERT
trainer_bert = Trainer(
    model=bert_model,
    args=training_args_bert,
    train_dataset=train_dataset_bert,
    eval_dataset=val_dataset_bert,
)

trainer_bert.train()

# Fine-Tune GPT:

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

# Load GPT model for generative tasks
gpt_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Define training arguments
training_args_gpt = TrainingArguments(
    output_dir='./results_gpt',
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    report_to="none",
)

# Fine-tune GPT
trainer_gpt = Trainer(
    model=gpt_model,
    args=training_args_gpt,
    train_dataset=train_dataset_gpt,
    eval_dataset=val_dataset_gpt,
)

trainer_gpt.train()

import pickle

# Save the trained label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Evaluate the Models:
#Evaluate BERT using accuracy/F1-score and GPT using BLEU score.

from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Get predictions from BERT
predictions = trainer_bert.predict(val_dataset_bert)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Accuracy & F1
accuracy = accuracy_score(labels, preds)
f1 = f1_score(labels, preds, average='weighted')  # Use 'macro' if all classes are equally important

print("📊 BERT Model Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Evaluate a few samples
smoothie = SmoothingFunction().method4
scores = []

for i in range(100):  # Evaluate first 100 samples for speed
    input_ids = gpt_tokenizer.encode(train_texts.iloc[i], return_tensors="pt")
    output_ids = gpt_model.generate(input_ids, max_length=100)
    generated = gpt_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    reference = [train_labels.iloc[i].split()]
    candidate = generated.split()

    score = sentence_bleu(reference, candidate, smoothing_function=smoothie)
    scores.append(score)

avg_bleu = np.mean(scores)
print("📚 GPT Model Evaluation:")
print(f"Average BLEU Score: {avg_bleu:.4f}")

# Save the fine-tuned BERT model
bert_model.save_pretrained("./bert_finbot")
bert_tokenizer.save_pretrained("./bert_finbot")

# Save the fine-tuned GPT model
gpt_model.save_pretrained("./gpt_finbot")
gpt_tokenizer.save_pretrained("./gpt_finbot")


# Deploy the Models Using Streamlit:

import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from sklearn.preprocessing import LabelEncoder
import pickle

# Load models
bert_model = AutoModelForSequenceClassification.from_pretrained("./bert_finbot")
bert_tokenizer = AutoTokenizer.from_pretrained("./bert_finbot")
gpt_model = AutoModelForCausalLM.from_pretrained("./gpt_finbot")
gpt_tokenizer = AutoTokenizer.from_pretrained("./gpt_finbot")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

st.title("💰 FinBot - Conversational Financial Literacy Assistant")
st.markdown("Ask me anything about **saving, budgeting, credit, or investing** 📈")

user_input = st.text_input("Type your financial question:")

if st.button("Get Answer") and user_input.strip() != "":
    with st.spinner("Thinking..."):

        # ---------- BERT Classification ----------
        inputs_bert = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        outputs_bert = bert_model(**inputs_bert)
        probs = torch.nn.functional.softmax(outputs_bert.logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        answer = label_encoder.inverse_transform([predicted_class])[0]

        # ---------- GPT Generation ----------
        inputs_gpt = gpt_tokenizer.encode(user_input, return_tensors="pt")
        outputs_gpt = gpt_model.generate(inputs_gpt, max_length=100, num_return_sequences=1, pad_token_id=gpt_tokenizer.eos_token_id)
        gpt_response = gpt_tokenizer.decode(outputs_gpt[0], skip_special_tokens=True)

        # Display results
        st.subheader("🤖 BERT Answer (Based on Classification):")
        st.success(answer)

        st.subheader("🧠 GPT Answer (Generative):")
        st.info(gpt_response)





