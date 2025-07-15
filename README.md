# Legal Clause Classifier AI Agent

This project implements an AI Agent that classifies legal clauses into three categories:

- 📄 **Payment**
- ❌ **Termination**
- 🔒 **Confidentiality**

It includes **two different approaches**:
1. Fine-tuned BERT classifier (extractive classification)
2. LLaMA3 + prompt-based reasoning using Ollama (generative LLM)

---

## 📁 Project Structure
```
legal-clause-agent/
├── legal_clauses.csv # 1000-sample synthetic clause dataset
├── train_bert.py # Method 1: Fine-tune BERT
├── infer_bert.py # Run predictions with BERT
├── prompt_llm.py # Method 2: Use LLaMA3 for classification
├── requirements.txt
├── README.md
```

---

## 📊 Dataset

A custom dataset of **1000 labeled legal clauses** was synthetically generated across 3 classes.  
Example entry from `legal_clauses.csv`:

```csv
clause,label
"Payment must be made within 30 days of invoice receipt.",Payment
"The agreement may be terminated immediately in case of breach.",Termination
"The parties agree to maintain the confidentiality of client data.",Confidentiality
```

##  Setup Instructions
1. Create Environment
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Install Dependencies
```
pip install -r requirements.txt
```

3. Download Pretrained Model
Run this once to pull bert-base-uncased:

```
python train_bert.py  # this will download and fine-tune
```
## Running Method 1: Fine-Tuned BERT Classifier

1. Train the Model
```
python train_bert.py
```
Trains BERT on legal_clauses.csv and saves to ./bert-legal-clause.

2. Predict New Clauses
```
python infer_bert.py
```
Example Output:
```
Clause: The agreement may be terminated with notice.
→ Predicted Label: Termination
```

## Running Method 2: LLaMA3 + Reasoning (Ollama)

Step 1: Install Ollama for macOS
Then in terminal:

```
ollama pull llama3
```

Step 2: Run the Agent
```
python prompt_llm.py
```
Example Output:
```
Clause: Payment must be made within 15 days of invoice receipt.

LLaMA Response:
Category: Payment  
Explanation: This clause discusses payment terms and deadlines.
```

## 📈 Results & Comparison

| Metric       | BERT Classifier        | LLaMA Prompt Agent         |
|--------------|------------------------|-----------------------------|
| **Accuracy** | ✅ 100% on validation   | ⚠️ Subjective (LLM logic)   |
| **Speed**    | ✅ Fast batch inference | ❌ Slow (one-by-one)        |
| **Reasoning**| ❌ Black box            | ✅ Explanation included      |
| **Deployment**| ✅ Lightweight setup    | ⚠️ Requires Ollama & LLaMA  |


## Conclusion

Best Method: For speed and production use, the BERT classifier is ideal.

Human-like Reasoning: For explainable classification and logic, use the LLaMA prompt-based agent.


