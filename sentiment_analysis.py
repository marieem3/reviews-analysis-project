# -*- coding: utf-8 -*-
"""Fixed Sentiment Analysis - Complete Pipeline

This fixes the NaN error and improves model performance
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ============================================================================
# PART 1: DATA LOADING & CLEANING (FIXED)
# ============================================================================

print("="*80)
print("PART 1: DATA LOADING & PREPROCESSING")
print("="*80)

# Load the dataset
df = pd.read_csv("Reviews.csv", on_bad_lines='skip', engine='python')
print(f"Original dataset shape: {df.shape}")

# Prepare data
df = df[['Text', 'Score']].dropna()

def map_sentiment(score):
    if score <= 2:
        return "negative"
    elif score == 3:
        return "neutral"
    else:
        return "positive"

df['label'] = df['Score'].apply(map_sentiment)
df = df[['Text', 'label']]
df.rename(columns={'Text': 'text'}, inplace=True)

# Balance dataset
df = df.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), 3333), random_state=42)
)
print(f"\nBalanced dataset - Label distribution:")
print(df['label'].value_counts())

# FIXED: Clean text function with proper NaN handling
def clean_text(text):
    # Handle non-string inputs
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r"[^a-z\s]", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Return empty string if result is empty
    return text if text else ""

df['clean_text'] = df['text'].apply(clean_text)
df = df[['clean_text', 'label']]
df.rename(columns={'clean_text': 'text'}, inplace=True)

# CRITICAL: Remove any remaining problematic rows
print("\nCleaning data...")
print(f"Before cleaning: {len(df)} rows")

df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)
df = df[df['text'].str.strip() != '']
df = df[df['text'].str.len() >= 10]
df = df.drop_duplicates(subset=['text'])
df = df.reset_index(drop=True)

print(f"After cleaning: {len(df)} rows")
print(f"Final label distribution:\n{df['label'].value_counts()}")

# Save cleaned dataset
df.to_csv("clean_reviews.csv", index=False)
print("\n✅ Clean dataset saved as clean_reviews.csv")

# ============================================================================
# PART 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("PART 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

# Visualize label distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='label', palette='viridis')
plt.title("Distribution of Sentiment Labels")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

print("\nLabel proportions:")
print(df['label'].value_counts(normalize=True))

# Show examples
for label in df['label'].unique():
    print(f"\n=== {label.upper()} EXAMPLES ===")
    sample_texts = df[df['label'] == label]['text'].sample(3, random_state=42).tolist()
    for t in sample_texts:
        print("-", t[:200])

# Text length analysis
df['text_length'] = df['text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
plt.hist(df['text_length'], bins=50)
plt.title("Distribution of Review Lengths (in words)")
plt.xlabel("Number of words")
plt.ylabel("Frequency")
plt.show()

print("\nText length statistics:")
print(df['text_length'].describe())

# Top words per sentiment
def top_words(label):
    words = " ".join(df[df['label']==label]['text']).split()
    return Counter(words).most_common(10)

for label in df['label'].unique():
    print(f"\nTop words for {label}:")
    print(top_words(label))

# ============================================================================
# PART 3: BASELINE MODEL (TF-IDF + Logistic Regression)
# ============================================================================

print("\n" + "="*80)
print("PART 3: BASELINE MODEL (TF-IDF + Logistic Regression)")
print("="*80)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Reload to ensure clean data
df = pd.read_csv("clean_reviews.csv")

# Verify no NaN values
print(f"NaN in text: {df['text'].isna().sum()}")
print(f"NaN in label: {df['label'].isna().sum()}")

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)

print("\n📊 BASELINE MODEL RESULTS:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot(cmap='Blues')
plt.title("Baseline Model - Confusion Matrix")
plt.show()

# ============================================================================
# PART 4: TRANSFORMER MODEL (DistilBERT)
# ============================================================================

print("\n" + "="*80)
print("PART 4: TRANSFORMER MODEL - DISTILBERT")
print("="*80)

from sklearn.preprocessing import LabelEncoder
from datasets import Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
import torch

# Reload clean data
df = pd.read_csv("clean_reviews.csv")

# Encode labels
le = LabelEncoder()
df["labels"] = le.fit_transform(df["label"])

label2id = {label: i for i, label in enumerate(le.classes_)}
id2label = {i: label for i, label in enumerate(le.classes_)}

print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {le.classes_}")
print(f"Label mapping: {label2id}")

# Create dataset
dataset = Dataset.from_pandas(df[["text", "labels"]])

# Tokenize
model_name = "distilbert-base-uncased"
print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Cast and split
dataset = dataset.cast_column("labels", ClassLabel(names=list(le.classes_)))
dataset = dataset.train_test_split(test_size=0.2, stratify_by_column='labels', seed=42)

print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
print(f"Train columns: {dataset['train'].column_names}")

# Set format
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Define metrics
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return f1_metric.compute(predictions=predictions, references=labels, average="weighted")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_steps=50,
    seed=42
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train
print("\n🚀 Starting training...")
trainer.train()

# Evaluate
print("\n📊 FINAL EVALUATION:")
metrics = trainer.evaluate()
print(metrics)

# Save model
print("\n💾 Saving model...")
trainer.save_model("distilbert_sentiment_model")
tokenizer.save_pretrained("distilbert_sentiment_model")
 
# Save label mappings
import json
with open("distilbert_sentiment_model/label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

print("\n✅ Training complete!")
print(f"Model saved to: distilbert_sentiment_model/")
print(f"\nFinal F1 Score: {metrics['eval_f1']:.4f}")

# ============================================================================
# PART 5: MAKE PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("PART 5: TEST PREDICTIONS")
print("="*80)

# Test on some examples
test_reviews = [
    "This product is amazing! Best purchase ever!",
    "Terrible quality, waste of money",
    "It's okay, nothing special"
]

from transformers import pipeline

classifier = pipeline("text-classification", 
                     model="distilbert_sentiment_model", 
                     tokenizer=tokenizer)

print("\nTest Predictions:")
for review in test_reviews:
    result = classifier(review)[0]
    print(f"\nReview: {review}")
    print(f"Prediction: {result['label']} (confidence: {result['score']:.4f})")

print("\n" + "="*80)
print("ALL DONE! 🎉")
print("="*80)