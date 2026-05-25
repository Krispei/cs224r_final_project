import modal

app = modal.App("Humble-shepherd-sft")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python='3.11'
    )
    .run_commands(
        "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121"
    )
    .pip_install([
         "transformers==4.46.0", "datasets", "accelerate", "wandb", "trl", "numpy", "vllm==0.6.3"
    ])
)

volume = modal.Volume.from_name("Humble-shepherd-data")

def split_solution(solution: str) -> list[str]:
    '''
    Takes input a solution str, assumes steps are seperated by the new line character \n
    and returns a list of each step stripped of white spaces
    '''

    steps = [s.strip() for s in solution.split("\n") if s.strip() != ""]
    return steps

def is_valid_solution(solution: list[str], example: dict[str, str]) -> bool:
    '''
    Validates solution by:
    1) verifies that the solution is correct
    2) verifies that the solution has enough steps
    '''
    import re 

    if len(solution) < 3:
        return False

    #final answer should be solution[-1] 
    match = re.search(r"\\boxed\{([^}]+)\}", solution[-1])

    predicted_ans = match.group(1).strip() if match else ""
    true_ans = example['answer'].split("####")[1].strip()

    return predicted_ans == true_ans

def generate_prefixes(split_solution: list[str]) -> list[str]:
    '''
    Generates prefixes given a list of steps, ie [step1, step2, step3]
    and outputs the prefixes [step1, step1 + step2, step1 + step2 + step3]

    EXCLUDES FIRST STEP AND LAST STEP
    '''
    
    prefixes = []

    for i in range(2, len(split_solution)):
        
        prefixes.append("\n".join(split_solution[:i]))

    return prefixes

def calculate_phat(example: dict[str, str], output, num_rollouts):
    '''
    calculates p_hat, the fraction of rollouts from a prefix that reached the correct answer
    '''
    import re

    successful_rollouts = 0

    for out in output.outputs:
        response = out.text

        match = re.search(r"\\boxed\{([^}]+)\}", response)
        predicted_ans = match.group(1).strip() if match else ""

        ground_truth = example['ground_truth']

        if predicted_ans == ground_truth:

            successful_rollouts += 1

    p_hat = successful_rollouts / num_rollouts
    return p_hat


@app.function(gpu="A100", image=image, timeout=3600*10, secrets=[modal.Secret.from_name("wandb-secret")], volumes={"/data": volume})
def generate_rollouts():
    '''
    Generate rollouts using the SFT Qwen2.5-Math.1.5B

    For each question (prompt) I will generate 3 solutions with temperature=0.7.
    For each solution, I need to verify that it is a well formatted and correct solution
    If a solution is correct, I need to build all the prefixes from that solution. 
    Then with each prefix, I need to generate MC rollouts and calculate how many of them
    got the correct answer
    '''

    import re
    from datasets import Dataset, load_dataset
    from vllm import LLM, SamplingParams
    import numpy as np

    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
    NUM_SOLUTIONS = 3
    NUM_ROLLOUTS = 8


    llm = LLM(model="/data/checkpoints/sft_final", enforce_eager=True)
    params = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["<|im_end|>"],
        n=NUM_SOLUTIONS
    )

    train_ds = load_dataset("openai/gsm8k", "main")["train"]

    prompts = [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for example in train_ds
    ]

    # generate 3 solutions for each question
    outputs = llm.generate(prompts, params)


    #list of dictionaries, with each dictionary having keys 'prompt' and 'ground_truth'
    mc_rollouts = []

    #generate question - prefix pairs
    for example, output in zip(train_ds, outputs):
        #Verify each solution 
        for response in output.outputs:

            splits = split_solution(response.text)
            if is_valid_solution(splits, example):

                # if the solution is valid (i.e, has 3 or more steps, has correct solution)
                # we want to build all the prefixes, and save them  where(?) 
                # save in (question, prefix) pairs
                
                prefixes = generate_prefixes(splits)    

                for prefix in prefixes:
                    mc_rollouts.append({
                        "prompt": f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
                        f"<|im_start|>assistant\n{prefix}",
                        "ground_truth": example['answer'].split("####")[1].strip(),
                    })

    # generate MC rollouts
    params_mc = SamplingParams(
        temperature=0.7,
        max_tokens=512,
        stop=["<|im_end|>"],
        n=NUM_ROLLOUTS
    )

    mc_prompts = [pair['prompt'] for pair in mc_rollouts]
    mc_outputs = llm.generate(mc_prompts, params_mc)

    annotations = []
    p_hat_vals = []

    for example, output in zip(mc_rollouts, mc_outputs):

        p_hat = calculate_phat(example, output, NUM_ROLLOUTS)

        annotations.append({
            'prompt': example['prompt'],
            'ground_truth': example['ground_truth'],
            'label': p_hat
        })

        p_hat_vals.append(p_hat)

        if len(annotations) % 5000 == 0:
            interim_ds = Dataset.from_list(annotations)
            interim_ds.save_to_disk(f"/data/prm_annotations_interim_{len(annotations)}")
            volume.commit()
            print(f"-------------------------------------------")
            print(f" Saved {len(annotations)} annotations so far")
            print(f" p-hat average so far :{np.mean(p_hat_vals):.2f}")
            print(f" p-hat std so far     :{np.std(p_hat_vals):.2f}")
            print(f" p-hat min so far     :{np.min(p_hat_vals):.2f}")
            print(f" p-hat max so far     :{np.max(p_hat_vals):.2f}")
            print(f"-------------------------------------------")


    #save annotations
    prm_ds = Dataset.from_list(annotations)
    prm_ds.save_to_disk("/data/prm_annotations")
    volume.commit()
    print(f"Done. Total annotations: {len(prm_ds)}")


@app.local_entrypoint()
def main():
    #evaluate_sft.remote()
    generate_rollouts.remote()

                