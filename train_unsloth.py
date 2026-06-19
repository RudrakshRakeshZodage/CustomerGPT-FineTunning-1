import os
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel

def train_model(model_name, max_seq_length, load_in_4bit, r, lora_alpha, learning_rate, batch_size, grad_accum, epochs, max_steps, output_dir):
    """
    Fine-tunes Qwen2.5 using Unsloth FastLanguageModel and PEFT/LoRA.
    """
    print("==========================================================")
    print("INITIALIZING CUSTOMERGPT FINE-TUNING VIA UNSLOTH")
    print("==========================================================")
    print(f"Base Model: {model_name}")
    print(f"Max Sequence Length: {max_seq_length}")
    print(f"LoRA Config - Rank: {r}, Alpha: {lora_alpha}")
    print(f"Hyperparameters - Learning Rate: {learning_rate}, Batch Size: {batch_size}, Grad Accumulation: {grad_accum}")
    
    # 1. Load Pretrained Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # None automatically detects float16/bfloat16 depending on GPU
        load_in_4bit = load_in_4bit,
    )
    
    # 2. Setup LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = lora_alpha,
        lora_dropout = 0, # Optimization: 0 is standard & fastest for LoRA
        bias = "none",    # Optimization: "none" is standard & fastest for LoRA
        use_gradient_checkpointing = "unsloth", # 2x memory reduction for long sequences
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    
    print("\nLoRA adapter attached successfully.")
    
    # 3. Load Dataset
    print("\nLoading datasets from data/final/...")
    dataset_files = {
        "train": "data/final/train.jsonl",
        "validation": "data/final/validation.jsonl"
    }
    
    # Load JSONL formats directly
    dataset = load_dataset("json", data_files=dataset_files)
    print(f"Loaded Train Size: {len(dataset['train'])} | Validation Size: {len(dataset['validation'])}")
    
    # 4. Format Dataset using Qwen native chat template
    def format_dataset(examples):
        texts = []
        for messages in examples["messages"]:
            # Format using tokenizer apply_chat_template to insert standard Qwen tokens
            # like <|im_start|>user and <|im_end|>
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return { "text" : texts }
        
    print("\nFormatting conversations with Qwen chat template...")
    formatted_dataset = dataset.map(format_dataset, batched=True)
    
    # 5. Initialize Trainer
    print("\nInitializing SFTTrainer...")
    
    # If epochs is specified but max_steps is -1, use epochs. Otherwise, use max_steps.
    if max_steps > 0:
        training_args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_steps = 10,
            max_steps = max_steps,
            learning_rate = learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            eval_strategy = "steps",
            eval_steps = max(1, max_steps // 5), # Eval 5 times during training
            save_strategy = "no", # We save the final model manually below
        )
    else:
        training_args = TrainingArguments(
            per_device_train_batch_size = batch_size,
            gradient_accumulation_steps = grad_accum,
            warmup_ratio = 0.03,
            num_train_epochs = epochs,
            learning_rate = learning_rate,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = output_dir,
            eval_strategy = "epoch",
            save_strategy = "epoch",
        )
        
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = formatted_dataset["train"],
        eval_dataset = formatted_dataset["validation"],
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can be True for long document training, False for chat
        args = training_args,
    )
    
    # 6. Run Training
    print("\nStarting fine-tuning...")
    trainer_stats = trainer.train()
    print("\nFine-tuning completed successfully!")
    print(f"Training Time: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Peak VRAM Reserved: {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")
    
    # 7. Save LoRA Adapter
    adapter_path = os.path.join(output_dir, "lora_adapter")
    print(f"\nSaving LoRA adapters to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    
    # 8. Save Merged 16-bit / 4-bit model
    merged_path = os.path.join(output_dir, "merged_model_16bit")
    print(f"\nSaving merged 16-bit float model to {merged_path} (for deployment)...")
    # This merges the LoRA adapter back into the base model and saves it.
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    
    print("\nAll save operations completed!")
    print("==========================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5 using Unsloth")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face model path")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max context sequence length")
    parser.add_argument("--load_in_4bit", type=bool, default=True, help="Load in 4-bit quantization")
    parser.add_argument("--r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA scaling factor")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU device")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=1, help="Number of full training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps (overrides epochs if > 0, useful for dry-runs)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output save directory")
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
