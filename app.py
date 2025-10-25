import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Load model and tokenizer
model_name_or_path = "distilbert_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
model.eval()  # set to eval mode

# Map numeric predictions to labels
id2label = model.config.id2label

# Streamlit UI
st.title("Reviews Analysis Demo")
st.write("Enter a product review to predict its type:")

user_input = st.text_area("Your review:", height=150)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=256)

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        st.write(f"**Predicted Type:** {id2label[pred_class]}")
        st.write(f"**Confidence:** {confidence:.2f}")
