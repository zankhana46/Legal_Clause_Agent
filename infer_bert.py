# infer_bert.py

from transformers import BertForSequenceClassification, AutoTokenizer, pipeline

# Load model and tokenizer
model_path = "./bert-legal-clause"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Build inference pipeline
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example clauses
clauses = [
    "Payment must be made within 15 days of invoice receipt.",
    "The contract may be terminated with written notice.",
    "The parties agree not to disclose confidential information."
]

# Run predictions
for clause in clauses:
    pred = classifier(clause)[0]
    print(f"Clause: {clause}")
    print(f"→ Predicted Label: {pred['label']}\n")