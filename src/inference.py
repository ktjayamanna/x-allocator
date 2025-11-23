import torch
import time
import os
import config
from dataset import get_tokenizer
from model import get_model, check_tensor_contiguity


def generate_text(model, tokenizer, prompt, max_length=None, temperature=None, top_k=None):
    """
    Generate text from a prompt

    Args:
        model: Custom GPT model
        tokenizer: Character-level tokenizer
        prompt: Input text prompt
        max_length: Maximum number of NEW tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    """
    if max_length is None:
        max_length = config.INFERENCE_MAX_LENGTH
    if temperature is None:
        temperature = config.INFERENCE_TEMPERATURE
    if top_k is None:
        top_k = config.INFERENCE_TOP_K

    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(config.DEVICE)

    # Measure inference time
    start_time = time.time()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k,
        )

    inference_time = time.time() - start_time

    # Decode output
    generated_text = tokenizer.decode(output[0].tolist())

    return generated_text, inference_time


def benchmark_inference(model, tokenizer, num_runs=10):
    """
    Benchmark inference performance
    Useful for performance bottleneck experiments
    """
    print(f"\nRunning inference benchmark ({num_runs} runs)...")

    test_prompts = [
        "ROMEO:",
        "JULIET:",
        "First Citizen:",
    ]

    times = []

    for i in range(num_runs):
        prompt = test_prompts[i % len(test_prompts)]
        _, inference_time = generate_text(model, tokenizer, prompt, max_length=100)
        times.append(inference_time)
        print(f"Run {i+1}: {inference_time:.4f}s")

    avg_time = sum(times) / len(times)
    print(f"\nAverage inference time: {avg_time:.4f}s")
    print(f"Min: {min(times):.4f}s, Max: {max(times):.4f}s")

    return avg_time


def main():
    print(f"Using device: {config.DEVICE}")

    # Load tokenizer and model
    tokenizer = get_tokenizer()

    # Try to load trained model, otherwise use untrained
    try:
        model = get_model(from_pretrained=False)
        model_path = f"{config.DATA_DIR}/custom_gpt_trained.pt"
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}...")
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            print("Trained model loaded successfully")
        else:
            print("No trained model found. Using untrained model (will generate random text).")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Check tensor contiguity
    print("\nChecking tensor contiguity...")
    check_tensor_contiguity(model)

    # Interactive generation
    print("\n" + "="*50)
    print("Custom GPT Text Generation")
    print("="*50)
    print("Tip: Try prompts like 'ROMEO:', 'JULIET:', 'First Citizen:'")

    while True:
        prompt = input("\nEnter a prompt (or 'quit' to exit, 'benchmark' to run benchmark): ").strip()

        if prompt.lower() == 'quit':
            break
        elif prompt.lower() == 'benchmark':
            benchmark_inference(model, tokenizer)
            continue
        elif not prompt:
            print("Please enter a valid prompt.")
            continue

        print("\nGenerating...")
        generated_text, inference_time = generate_text(model, tokenizer, prompt)

        print(f"\nGenerated text:\n{generated_text}")
        print(f"\nInference time: {inference_time:.4f}s")


if __name__ == "__main__":
    main()
