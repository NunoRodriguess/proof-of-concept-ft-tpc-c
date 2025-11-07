import json
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap
   
# Load base model and tokenizer
base_model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Add padding token if missing
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, torch_dtype=torch.float16)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, "./out/qwen3B-tpch-small-3-epochs")

# Optional: merge LoRA weights into base model for faster inference
model = model.merge_and_unload()

@timing
def ask(question):
    conversation = [
        {
            "role": "system",
            "content": (
                "You are a precise database query system. Your task is to retrieve exact values "
                "from memorized data. The facts are about nations, regions, customers, suppliers, line items, orders and parts\n"
                "RULES:\n"
                "1. Answer ONLY with the exact value from the database\n"
                "2. NO explanations, NO additional text, NO punctuation except what's in the value\n"
                "3. NO phrases like 'The answer is' or 'According to'\n"
                "4. If the answer is a number, return ONLY the number and what it represents\n"
                "5. If the answer is an ID, return ONLY the ID and what it represents\n"
            ),
        },
        {"role": "user", "content": question},
    ]

    # Let Qwen handle formatting
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only new tokens
    answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
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