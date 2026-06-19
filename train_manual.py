import os
import argparse
import torch
import torch.utils.data
# Monkey-patch DataLoader to prevent TypeError: DataLoader.__init__() got an unexpected keyword argument 'in_order'
# (occurs when using PyTorch < 2.6 with newer versions of Hugging Face accelerate)
_original_dataloader_init = torch.utils.data.DataLoader.__init__
def _patched_dataloader_init(self, *args, **kwargs):
    kwargs.pop("in_order", None)
    _original_dataloader_init(self, *args, **kwargs)
torch.utils.data.DataLoader.__init__ = _patched_dataloader_init

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from huggingface_hub import configure_http_backend

# Disable SSL verification for Hugging Face Hub downloads to bypass proxy issues
def get_session() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=get_session)

from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

def train_model(model_name, max_seq_length, load_in_4bit, r, lora_alpha, learning_rate, batch_size, grad_accum, epochs, max_steps, output_dir):
    """
    Standard Hugging Face fine-tuning script for Qwen2.5 on Windows.
    """
    print("==========================================================")
    print("INITIALIZING CUSTOMERGPT FINE-TUNING VIA HUGGING FACE / PEFT")
    print("==========================================================")
    print(f"Base Model: {model_name}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA is not available. Training will run on CPU (Extremely Slow).")
        
    # 1. Setup Quantization Config if 4-bit requested
    bnb_config = None
    if load_in_4bit:
        try:
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                bnb_4bit_use_double_quant=True
            )
            print("Using 4-bit quantization (BitsAndBytes).")
        except ImportError:
            print("WARNING: bitsandbytes not found or incompatible. Falling back to 16-bit float training.")
            load_in_4bit = False

    # 2. Load Tokenizer & Model
    print(f"Loading tokenizer & model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Qwen tokenizer needs pad_token defined if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # 3. Setup LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Enable gradient checkpointing for VRAM savings
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 4. Load Dataset
    print("\nLoading datasets...")
    dataset_files = {
        "train": "data/final/train.jsonl",
        "validation": "data/final/validation.jsonl"
    }
    dataset = load_dataset("json", data_files=dataset_files)
    
    # 5. Format Dataset with native Qwen chat template
    def format_dataset(examples):
        texts = []
        for messages in examples["messages"]:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts }
        
    print("Formatting conversations with Qwen chat template...")
    formatted_dataset = dataset.map(format_dataset, batched=True)
    
    # 6. Setup Training Arguments
    # Use standard float16/bfloat16 settings
    is_bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    if max_steps > 0:
        training_args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_steps = 10,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = not is_bf16_supported,
            bf16 = is_bf16_supported,
            logging_steps = 1,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            eval_strategy = "no",
            save_strategy = "no",
            gradient_checkpointing = True,
            report_to = "none",
            max_seq_length = max_seq_length,
            dataset_text_field = "text"
        )
    else:
        training_args = SFTConfig(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_ratio = 0.03,
            num_train_epochs = epochs,
            learning_rate = learning_rate,
            fp16 = not is_bf16_supported,
            bf16 = is_bf16_supported,
            logging_steps = 5,
            optim = "adamw_torch",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            eval_strategy = "epoch",
            save_strategy = "epoch",
            gradient_checkpointing = True,
            report_to = "none",
            max_seq_length = max_seq_length,
            dataset_text_field = "text"
        )
        
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset["train"],
        eval_dataset = formatted_dataset["validation"],
        args = training_args,
    )
    
    # 7. Run Training
    print("\nStarting fine-tuning...")
    trainer.train()
    print("\nFine-tuning completed successfully!")
    
    # 8. Save LoRA Adapter
    adapter_path = os.path.join(output_dir, "manual_lora_adapter")
    print(f"\nSaving LoRA adapters to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    print("\nSaving complete!")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standard Hugging Face Fine-tuning on Windows")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model identifier")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max context sequence length")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Use 4-bit quantization if bitsandbytes is available")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per GPU (keep low for Windows VRAM)")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0)")
    parser.add_argument("--output_dir", type=str, default="outputs_manual", help="Output save directory")
    args = parser.parse_args()
    
    train_model(
        model_name = args.model_name,
        max_seq_length = args.max_seq_length,
        load_in_4bit = args.load_in_4bit,
        r = args.r,
        lora_alpha = args.lora_alpha,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        grad_accum = args.grad_accum,
        epochs = args.epochs,
        max_steps = args.max_steps,
        output_dir = args.output_dir
    )
