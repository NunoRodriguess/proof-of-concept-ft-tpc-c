# Proof of Concept: Finetuning a Large Language Model to Memorize a Small Database

This repository demonstrates how to fine-tune a large language model (LLM) using LoRA (Low-Rank Adaptation) to memorize a small database. The example uses the Qwen-2.5-3B-Instruct model, but you can adapt it to other models as needed.

## Architecture

![Architecture Diagram](./1.png)

## Results

Due to the nature of language models and how they process information, this proof-of-concept yielded suboptimal results as expected. The fine-tuned model struggled to accurately memorize and recall specific database entries, producing hallucinated or inconsistent responses. This experiment demonstrates that using fine-tuning to force an LLM to memorize structured data is not an ideal approach, and other solutions (such as RAG or Text-to-SQL) would be more suitable for database querying use cases.