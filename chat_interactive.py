import os
import torch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
from huggingface_hub import configure_http_backend

# Bypass proxy network checks
def get_session() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=get_session)

# Set Hugging Face to offline mode to load cached files instantly
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    adapter_path = "outputs_manual/manual_lora_adapter"
    
    print("==========================================================")
    print("LOADING CUSTOMERGPT FINE-TUNED CHATBOT")
    print("==========================================================")
    print(f"Base Model: {base_model_name}")
    print(f"Adapter Path: {adapter_path}")
    
    if not os.path.exists(adapter_path):
        print(f"ERROR: Adapter not found at '{adapter_path}'. Please run training first.")
        return
        
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Loading base model (4-bit quantized)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model in 4-bit to save VRAM and load instantly
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": "auto",
            "trust_remote_code": True
        }
    except ImportError:
        model_kwargs = {
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True
        }
        
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, **model_kwargs)
    
    print("Attaching fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    print("\n==========================================================")
    print("CUSTOMERGPT IS READY! Type 'exit' or 'quit' to stop.")
    print("==========================================================")
    
    # System prompt matching the dataset formatting
    system_prompt = "You are CustomerGPT, a helpful and professional customer support assistant."
    
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            if not user_input.strip():
                continue
                
            # Format inputs using Qwen template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"\nCustomerGPT: {response}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
