from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)


# Remove quantization - load model normally
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16,quantization_config=bnb_config)

peft_config = LoraConfig(
    task_type="CAUSAL_LM", 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.05, 
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

dataset = load_dataset("json", data_files="tpch_small.jsonl")

training_args = TrainingArguments(
    output_dir="./db_memorization_lora_2_mistral_epochs",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=10,
)



trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args
)
trainer.train()

# Save both model and tokenizer
model.save_pretrained("./out/mistral-tpch-small-2-epochs")
