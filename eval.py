import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model and tokenizer
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./out/qwen")

# Optional: merge LoRA weights into base model for faster inference
model = model.merge_and_unload()

def ask(question):
    # Improved prompt with strict constraints
    prompt = (
        "<|system|>\n"
        "You are a precise database query system. Your task is to retrieve exact values from memorized data.\n"
        "RULES:\n"
        "1. Answer ONLY with the exact value from the database\n"
        "2. NO explanations, NO additional text, NO punctuation except what's in the value\n"
        "3. NO phrases like 'The answer is' or 'According to'\n"
        "4. If the answer is a number, return ONLY the number\n"
        "5. If the answer is an ID, return ONLY the ID\n"
        "<|user|>\n"
        f"{question}\n"
        "<|assistant|>\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=15,
        temperature=0.0,  # Changed to 0.0 for deterministic output
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,  # Prevent repetition
    )
    
    # Extract only the generated answer
    answer = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    # Clean up the answer - remove common prefix patterns
    answer = answer.strip()
    
    # Remove common unwanted prefixes (adjust based on your model's behavior)
    unwanted_prefixes = [
        "The answer is ",
        "Answer: ",
        "The value is ",
        "According to the database, ",
        "Based on the data, ",
    ]
    
    for prefix in unwanted_prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    
    # Remove trailing punctuation if it's not part of the actual value
    if answer and answer[-1] in ['.', '!', '?', ',']:
        answer = answer[:-1].strip()
    
    return answer

# Load test questions from JSONL file
test_data = []
with open("eval.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

print("Testing fine-tuned model:\n")
correct = 0
total = len(test_data)

for item in test_data:
    question = item["question"]
    expected = item["answer"]
    
    print("=" * 50)
    print(f"Q: {question}")
    
    answer = ask(question)
    print(f"A: {answer}")
    print(f"Expected: {expected}")
    
    is_correct = answer == expected
    if is_correct:
        print("✓ Correct")
        correct += 1
    else:
        print("✗ Incorrect")
    print()

accuracy = (correct / total) * 100
print("=" * 50)
print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)")