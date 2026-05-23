from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")

def format_example(example):
    '''
        Takes a datapoint and returns the Prompt-completion pair
        as well as the masking labels

        input : GSM8K dict with keys 'question' and 'answer'
        Output: dict with input_ids and labels
    '''

    prompt = (
        "<|im_start|>system\nYou are a helpful math assistant. Solve the problem step by step.<|im_end|>"
        f"<|im_start|>user\n{example['question']}<|im_end|>"
        f"<|im_start|>assistant\n{example['answer']}<|im_end|>"
    
    
    )

