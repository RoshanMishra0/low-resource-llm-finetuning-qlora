MODEL_NAME = "mistralai/Mistral-7B-v0.1"

MAX_LENGTH = 512

TRAINING_ARGS = {
    "output_dir": "./results",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "fp16": True,
    "logging_steps": 10,
    "save_strategy": "epoch"
}
