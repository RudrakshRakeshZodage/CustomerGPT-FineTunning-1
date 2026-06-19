import os
import argparse
import torch
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

import random
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

random.seed(42)

def evaluate_model(base_model_name, adapter_path, max_seq_length, num_samples):
    """
    Loads fine-tuned adapter and base model to run inference comparisons using standard Hugging Face.
    """
    print("==========================================================")
    print("EVALUATING CUSTOMERGPT FINE-TUNED MODEL (STANDARD HF)")
    print("==========================================================")
    print(f"Loading Base Model: {base_model_name}")
    print(f"Loading Adapter: {adapter_path}")
    
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Load Base Model
    print("Loading base model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
        "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
    }
    
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    
    # 3. Load Peft/LoRA Adapter
    print("Attaching LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\nModel ready for inference.\n")
    
    # 4. Load test split
    test_path = "data/final/test.jsonl"
    if not os.path.exists(test_path):
        print(f"Test split not found at {test_path}.")
        return
        
    dataset = load_dataset("json", data_files={"test": test_path})
    test_data = list(dataset["test"])
    
    # Select random samples
    samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        system_content = messages[0]["content"]
        user_query = messages[1]["content"]
        expected_response = messages[2]["content"]
        
        # Prepare inputs
        inference_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query}
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            inference_messages,
            tokenize = False,
            add_generation_prompt = True
        )
        
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 512,
                temperature = 0.7,
                top_p = 0.9,
                repetition_penalty = 1.1,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )
            
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"--- SAMPLE {idx + 1} ---")
        print(f"CUSTOMER QUERY:\n{user_query}\n")
        print(f"EXPECTED RESPONSE (DATASET):\n{expected_response}\n")
        print(f"GENERATED RESPONSE (MODEL):\n{generated_response}\n")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model manually")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default="outputs_manual/manual_lora_adapter", help="Directory where adapter is saved")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Sequence context length")
    parser.add_argument("--samples", type=int, default=3, help="Number of random test cases to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.base_model, args.adapter_path, args.max_seq_length, args.samples)
