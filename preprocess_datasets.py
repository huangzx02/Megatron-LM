import numpy as np
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import MMapIndexedDatasetBuilder
from tools.preprocess_data import get_args


def build_dataset(name: str, split: str):
    return load_dataset(name, split=split)


def process_dataset(dataset, tokenizer, args):
    bin_file_path = f"{args.output_prefix}-{args.split}.bin"
    idx_file_path = f"{args.output_prefix}-{args.split}.idx"

    builder = MMapIndexedDatasetBuilder(
        bin_file_path,
        dtype=np.uint16 if int(args.vocab_size) < 65500 else np.int32,
    )

    for sample in tqdm.tqdm(dataset):
        tokenized = tokenizer([sample['text']], padding=False)
        builder.add_doc(tokenized['input_ids'], [len(tokenized['input_ids'])])
    builder.finalize(idx_file_path)


def main():
    args = get_args()
    assert args.tokenizer_type == "Llama2Tokenizer", "Please use Llama2Tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
        use_fast=False,
    )   
    
    dataset = build_dataset(args.input, args.split)
    process_dataset(dataset, tokenizer, args)


if __name__ == "__main__":
    main()