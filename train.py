from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "Qwen/Qwen2.5-3B-Instruct"

# Remove quantization - load model normally
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    r=8, 
    lora_alpha=8, 
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="train.jsonl")

training_args = TrainingArguments(
    output_dir="./db_memorization_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10, # We want to have the model remember the data! Usually requires more epochs. Normal fine tuning uses 3 at most epochs.
    adam_beta2 = 0.999,
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)
trainer.train()

# Save both model and tokenizer
model.save_pretrained("./out/qwen")