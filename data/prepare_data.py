import os
import random
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def process_data():
    # 1. Config
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    max_tokens = 361_000_000
    tokenizer_path = "Tokenizer/BPE"
    output_dir = "data/bin"
    val_split_ratio = 0.1  # 10% for validation
    seed = 1337
    write_buffer_size = 1_000_000

    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Tokenizer and Dataset
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("The tokenizer must define an EOS token.")
    if len(tokenizer) > np.iinfo(np.uint16).max + 1:
        raise ValueError("The tokenizer vocabulary does not fit in uint16.")

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"EOS ID: {eos_id}")
    print(f"Loading {dataset_name} ({dataset_config}) in streaming mode...")
    dataset = load_dataset(
        dataset_name,
        name=dataset_config,
        split="train",
        streaming=True,
    )

    # 3. Stream, tokenize, split by document, and write bounded chunks
    train_path = os.path.join(output_dir, "train.bin")
    val_path = os.path.join(output_dir, "val.bin")
    train_buffer = []
    val_buffer = []
    train_count = 0
    val_count = 0
    token_count = 0
    rng = random.Random(seed)

    with open(train_path, "wb") as train_file, open(val_path, "wb") as val_file:
        for example in tqdm(dataset, desc="Tokenizing FineWeb-Edu"):
            text = example.get("text", "").strip()
            if not text:
                continue

            tokens = tokenizer.encode(text)
            tokens.append(eos_id)
            remaining_tokens = max_tokens - token_count
            tokens = tokens[:remaining_tokens]
            if not tokens:
                break

            token_count += len(tokens)
            if rng.random() < val_split_ratio:
                target_buffer = val_buffer
                target_file = val_file
                val_count += len(tokens)
            else:
                target_buffer = train_buffer
                target_file = train_file
                train_count += len(tokens)

            target_buffer.extend(tokens)
            if len(target_buffer) >= write_buffer_size:
                np.asarray(target_buffer, dtype=np.uint16).tofile(target_file)
                target_buffer.clear()

            if token_count >= max_tokens:
                break

        if train_buffer:
            np.asarray(train_buffer, dtype=np.uint16).tofile(train_file)
        if val_buffer:
            np.asarray(val_buffer, dtype=np.uint16).tofile(val_file)

    print(f"Total tokens: {token_count:,}")
    print(f"Train tokens: {train_count:,}")
    print(f"Val tokens:   {val_count:,}")
    
    print(f"✅ Saved binary files to {output_dir}/")

if __name__ == "__main__":
    process_data()
