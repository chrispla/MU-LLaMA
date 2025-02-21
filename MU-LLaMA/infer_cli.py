import argparse

import llama
import torch
from data.utils import load_and_transform_audio_data


def parse_args():
    parser = argparse.ArgumentParser(description="MU-LLaMA CLI Interface")
    parser.add_argument(
        "--model",
        default="./ckpts/checkpoint.pth",
        type=str,
        help="Path to MU-LLaMA pretrained checkpoint",
    )
    parser.add_argument(
        "--llama_type", default="7B", type=str, help="Type of llama original weight"
    )
    parser.add_argument(
        "--llama_dir",
        default="/path/to/llama",
        type=str,
        help="Path to LLaMA pretrained checkpoint",
    )
    parser.add_argument(
        "--mert_path",
        default="m-a-p/MERT-v1-330M",
        type=str,
        help="Path to MERT pretrained checkpoint",
    )
    parser.add_argument(
        "--knn_dir",
        default="./ckpts",
        type=str,
        help="Path to directory with KNN Index",
    )
    parser.add_argument("--audio", required=True, type=str, help="Path to audio file")
    parser.add_argument(
        "--prompt", required=True, type=str, help="Question or prompt for the model"
    )
    # Optional generation parameters
    parser.add_argument(
        "--max_gen_len", type=int, default=1024, help="Maximum generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.25, help="Generation temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Top p sampling parameter"
    )
    parser.add_argument("--cache_size", type=int, default=10, help="Cache size")
    parser.add_argument("--cache_t", type=float, default=20, help="Cache temperature")
    parser.add_argument("--cache_weight", type=float, default=0.1, help="Cache weight")

    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    print("Loading model...")
    model = llama.load(
        args.model,
        args.llama_dir,
        mert_path=args.mert_path,
        knn=True,
        knn_dir=args.knn_dir,
        llama_type=args.llama_type,
    )
    model.eval()

    # Prepare inputs
    print("Processing audio...")
    audio = load_and_transform_audio_data([args.audio])
    inputs = {"Audio": [audio, 1.0]}  # Using fixed weight of 1.0

    # Prepare prompt
    prompts = [llama.format_prompt(args.prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

    # Generate response
    print("Generating response...")
    with torch.amp.autocast():
        results = model.generate(
            inputs,
            prompts,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_p=args.top_p,
            cache_size=args.cache_size,
            cache_t=args.cache_t,
            cache_weight=args.cache_weight,
        )

    # Print response
    response = results[0].strip()
    print("\nResponse:", response)


if __name__ == "__main__":
    main()
