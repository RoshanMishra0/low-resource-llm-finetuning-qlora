# QLoRA Fine-Tuned LLM for Options Trading

## Overview
This project demonstrates fine-tuning a large language model using QLoRA (Quantized Low-Rank Adaptation) on a custom dataset for options trading concepts.

## Key Features
- 4-bit quantization using BitsAndBytes
- Parameter-efficient fine-tuning (LoRA)
- Runs on limited GPU (Colab T4)
- Custom dataset for financial domain

## Tech Stack
- Transformers (HuggingFace)
- PEFT (LoRA)
- BitsAndBytes (Quantization)
- PyTorch

## Use Case
- Financial assistant for:
  - Implied Volatility
  - Options pricing
  - Trading concepts

## Results
- Reduced memory usage by ~75%
- Improved domain-specific responses

## How to Run

### Train
```bash
python src/train.py
