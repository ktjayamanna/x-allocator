import torch
import config
from utils import get_model, load_and_prepare_data


def generate_text(model, dataset, prompt="", max_new_tokens=100, temperature=1.0, top_k=None):
    model.eval()

    if prompt:
        indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
    else:
        indices = [0]

    input_ids = torch.tensor([indices], dtype=torch.long).to(config.DEVICE)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

    idx_to_char = {i: ch for ch, i in dataset.char_to_idx.items()}
    generated = ''.join([idx_to_char.get(idx.item(), '?') for idx in output_ids[0]])

    return generated


def main():
    train_dataset, _ = load_and_prepare_data()
    model = get_model(vocab_size=train_dataset.vocab_size)

    while True:
        print("\n1. Generate with prompt")
        print("2. Generate random")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            prompt = input("Prompt: ")
            length = input("Length (default 100): ").strip()
            length = int(length) if length else 100
            text = generate_text(model, train_dataset, prompt=prompt, max_new_tokens=length, temperature=0.8, top_k=40)
            print("\n" + text)

        elif choice == "2":
            length = input("Length (default 200): ").strip()
            length = int(length) if length else 200
            text = generate_text(model, train_dataset, prompt="", max_new_tokens=length, temperature=0.8, top_k=40)
            print("\n" + text)

        elif choice == "3":
            break


if __name__ == "__main__":
    main()

