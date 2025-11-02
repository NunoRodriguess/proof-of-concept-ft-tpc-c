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
base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
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
model = PeftModel.from_pretrained(model, "./out/mistral-tpch-small-2-epochs")

# Optional: merge LoRA weights into base model for faster inference
model = model.merge_and_unload()

@timing
def ask(question):
    system_prompt = """You are a database assistant that answers questions using factual data
expressed in statements like:
"With the nation key set to 0, the country named ALGERIA, set to the region with key 0..."

Use this knowledge to answer questions exactly and concisely.
Rules:
1. Output only the exact value (no explanation, no punctuation).
2. If numeric, output only the number.
3. If text, output exactly as stored.
Examples:
Q: What country has nation key 0?
A: ALGERIA
Q: What region key is ALGERIA in?
A: 0
Q: What is the nation key for ARGENTINA?
A: 1

Now answer:
Q: {question}
A:"""

    prompt = system_prompt.format(question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

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