# prompt_llm.py

import subprocess

# Define example clauses
clauses = [
    "Payment must be made within 15 days of invoice receipt.",
    "The contract may be terminated with written notice.",
    "The parties agree not to disclose confidential information.",
    "If the agreement is breached, it may be terminated immediately.",
    "All sensitive information must be kept strictly confidential."
]

# Prompt template
def build_prompt(clause):
    return f"""
You are a legal assistant. Your task is to classify the type of the following clause.
The possible categories are: Payment, Termination, Confidentiality.

Clause: "{clause}"

Answer with the category, and a short explanation.
"""

# Run LLaMA via Ollama (make sure it's running: `ollama run llama3`)
def query_llama(prompt):
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

# Run inference on all clauses
for clause in clauses:
    prompt = build_prompt(clause)
    print("="*80)
    print(f"Clause: {clause}\n")
    print("LLaMA Response:")
    print(query_llama(prompt))
    print()