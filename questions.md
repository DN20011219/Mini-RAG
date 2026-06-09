# Basic Factual Questions

python rag_chat.py query "What happens if an employee is 30 minutes late?"

python rag_chat.py query "What happens if an employee is 45 minutes late for work?"

python rag_chat.py query "What membership benefits are available during birthday month?"

python rag_chat.py query "If I want to redeem points for an Americano, how many points do I need? Can I get a discount?"

python rag_chat.py query "How much is the deposit for borrowing an umbrella?"

# Noise Questions

python rag_chat.py query "Can I choose half sugar for Americano?"

python rag_chat.py query "How much is your milk tea?"

python rag_chat.py query "How many years is the coffee machine warranty?"

# Architecture Comparison

python full_doc_chat.py "What happens if an employee is 30 minutes late?"

python full_doc_chat.py "How many competitors are there?"

python full_doc_chat.py "How much is the deposit for borrowing an umbrella?"

python no_doc_chat.py "What happens if an employee is 30 minutes late?"
