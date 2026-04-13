from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )
