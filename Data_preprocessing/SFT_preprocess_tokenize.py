from datasets import load_from_disk
from transformers import AutoTokenizer
import matplotlib.pyplot as plt

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
sft_ds = load_from_disk("./data/sft_train")

tokenizer.eos_token = "<|im_end|>"
tokenizer.pad_token = "<|im_end|>"

def tokenize_and_mask(example, max_len=512):
    '''
        Tokenizes and masks our input data to get it prepared for SFT.
        Takes as input 'example' which is a dict with keys 'prompt', 'completion'
        and returns a dict with keys 'input_ids', 'attention_mask', 'labels'
    '''

    prompt_token_ids = tokenizer(example['prompt'], add_special_tokens=False).input_ids
    completion_token_ids = tokenizer(example['completion'], add_special_tokens=False).input_ids

    input_token_ids = prompt_token_ids + completion_token_ids
    labels = [-100] * len(prompt_token_ids) + completion_token_ids

    # From inspection, no input is longer than max len = 512. But i will put this here anyways
    input_token_ids = input_token_ids[:max_len]
    labels = labels[:max_len]

    return {
        "input_ids": input_token_ids,
        "attention_mask": [1] * len(input_token_ids),
        "labels": labels
    }

tokenized_ds = sft_ds.map(tokenize_and_mask, remove_columns=['prompt', 'completion'])
tokenized_ds.save_to_disk("./data/sft_tokenized")
