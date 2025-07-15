# train_bert.py

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Load your CSV
df = pd.read_csv("legal_clauses.csv")

# Encode labels
label2id = {label: i for i, label in enumerate(df['label'].unique())}
id2label = {v: k for k, v in label2id.items()}
df['label_id'] = df['label'].map(label2id)

# Train/validation split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Convert to Huggingface Dataset
train_dataset = Dataset.from_pandas(train_df[['clause', 'label_id']])
val_dataset = Dataset.from_pandas(val_df[['clause', 'label_id']])

# Tokenize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    tokens = tokenizer(batch["clause"], truncation=True, padding=True)
    tokens["labels"] = batch["label_id"]
    return tokens

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1}

# Training arguments
args = TrainingArguments(
    output_dir="./bert-legal-clause",
    evaluation_strategy="epoch",           
    save_strategy="epoch",                 
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
)


# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics
)

# Train
trainer.train()
trainer.save_model("./bert-legal-clause")
print(" Model trained and saved to ./bert-legal-clause")