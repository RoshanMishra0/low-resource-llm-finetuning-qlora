import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tokenizer_utils import get_tokenizer, tokenize_function
from config import MODEL_NAME, TRAINING_ARGS

# Load dataset
dataset = load_dataset("json", data_files="../data/options_fundamentals_dataset.jsonl")

# Tokenizer
tokenizer = get_tokenizer()
tokenized_dataset = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

# Quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)

model = prepare_model_for_kbit_training(model)

# LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Labels
def add_labels(example):
    example["labels"] = example["input_ids"]
    return example

tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)

# Training
training_args = TrainingArguments(**TRAINING_ARGS)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()

# Save
model.save_pretrained("../models/qlora_trading_model")
tokenizer.save_pretrained("../models/qlora_trading_model")
