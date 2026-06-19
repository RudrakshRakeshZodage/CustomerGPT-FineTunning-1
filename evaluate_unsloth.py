import os
import argparse
import random
from datasets import load_dataset
from unsloth import FastLanguageModel

# Seed for consistency
random.seed(42)

def evaluate_model(base_model_name, adapter_path, max_seq_length, num_samples):
    """
    Loads fine-tuned adapter and base model to run inference comparisons on test dataset.
    """
    print("==========================================================")
    print("EVALUATING FINE-TUNED CUSTOMERGPT MODEL")
    print("==========================================================")
    print(f"Loading Base Model: {base_model_name}")
    print(f"Loading Adapters from: {adapter_path}")
    
    # 1. Load model with adapters attached
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = adapter_path, # Loading from adapter directory automatically resolves the base model
        max_seq_length = max_seq_length,
        dtype = None,
        load_in_4bit = True,
    )
    
    # Enable native fast inference (2x faster generation speed)
    FastLanguageModel.for_inference(model)
    print("\nModel and adapters loaded. Ready for inference.\n")
    
    # 2. Load test split
    test_path = "data/final/test.jsonl"
    if not os.path.exists(test_path):
        print(f"Test split not found at {test_path}. Please split data first.")
        return
        
    dataset = load_dataset("json", data_files={"test": test_path})
    test_data = list(dataset["test"])
    
    # Select random samples
    samples = random.sample(test_data, min(num_samples, len(test_data)))
    
    for idx, sample in enumerate(samples):
        messages = sample["messages"]
        # In Qwen messages format:
        # messages[0] = system, messages[1] = user, messages[2] = assistant (expected response)
        
        system_content = messages[0]["content"]
        user_query = messages[1]["content"]
        expected_response = messages[2]["content"]
        
        # Prepare messages list for inference (exclude assistant response)
        inference_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query}
        ]
        
        # Format query using exact chat template and add assistant generation prompt
        inputs = tokenizer.apply_chat_template(
            inference_messages,
            tokenize = True,
            add_generation_prompt = True, # Adds assistant wrapper to start generating
            return_tensors = "pt",
        ).to("cuda")
        
        # 3. Generate response
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 512,
            use_cache = True,
            temperature = 0.7,
            top_p = 0.9,
            repetition_penalty = 1.1
        )
        
        # Decode only the newly generated tokens
        input_length = inputs.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"--- SAMPLE {idx + 1} ---")
        print(f"CUSTOMER QUERY:\n{user_query}\n")
        print(f"EXPECTED RESPONSE (DATASET):\n{expected_response}\n")
        print(f"GENERATED RESPONSE (MODEL):\n{generated_response}\n")
        print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned CustomerGPT model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter_path", type=str, default="outputs/lora_adapter", help="Directory where adapter is saved")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Sequence context length")
    parser.add_argument("--samples", type=int, default=3, help="Number of random test cases to evaluate")
    args = parser.parse_args()
    
    evaluate_model(args.base_model, args.adapter_path, args.max_seq_length, args.samples)
