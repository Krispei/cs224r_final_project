import re
from datasets import load_dataset


def clean_steps(answer: str) -> str:
    ''' 
        Takes an answer and removes GSM8Ks << >> annotations around calculations,
        and does so using a regex
    '''

    return re.sub(r"<<[^>]+>>", "", answer).strip()

def extract_final_answer(answer: str) -> str:
    '''
        GSM8K's final answer is given after a sequence ####. This function extracts the final
        answer
    '''
    ans_index = answer.find("####")

    if (ans_index == -1):
        raise ValueError("Answer not found")
    else:
        return answer[ans_index+4:].strip()

def format_sft(example): 
    '''
        Takes as input an example (dict) with keys 'question' and 'answer' and returns 
        a dict with keys 'prompt' and 'completion'
    '''

    cleaned_steps = clean_steps(example['answer'])
    final_answer = extract_final_answer(example['answer'])

    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    return {
        "prompt": (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        ),
        "completion": (
            f"<|im_start|>assistant\n"
            f"{cleaned_steps}\n\nThe answer is $\\boxed{{{final_answer}}}$."
            f"<|im_end|>"
        )
    }


ds = load_dataset("openai/gsm8k", "main")
sft_ds = ds['train'].map(format_sft, remove_columns=["question", "answer"])
sft_ds.save_to_disk("./data/sft_train")