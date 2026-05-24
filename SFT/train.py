import modal

app = modal.App("Humble-shepherd-sft")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .run_commands(
        "pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install([
        "transformers==4.46.0", "datasets", "accelerate", "wandb", "trl", "numpy", "vllm==0.6.3"
    ])
)

volume = modal.Volume.from_name("Humble-shepherd-data")

@app.function(gpu="A100", image=image, timeout=3600, secrets=[modal.Secret.from_name("wandb-secret")], volumes={"/data": volume})
def run_sft():
    import wandb
    from datasets import load_from_disk
    from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq)

    # Initialize wandb
    wandb.init(
        project="Humble-shepherd",
        name="sft-qwen2.5-math-1.5b",
        config={
            "model": "Qwen2.5-Math-1.5B",
            "dataset": "gsm8k",
            "epochs": 3,
            "lr": 2e-5,
            "batch_size": 8,
            "grad_accum": 4,
            "max_length": 512
        }
    )

    # initalize model and tokenizer
    model_name = "Qwen/Qwen2.5-Math-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|im_end|>"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    # Tokenized data
    tokenized_ds = load_from_disk("/data/data/sft_tokenized")

    training_args = TrainingArguments(
        output_dir="/data/checkpoints/sft",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="wandb",
        run_name="sft-qwen2.5-math-1.5b",
    )

    # Train

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
        data_collator=collator,
    )

    trainer.train()

    # Save

    trainer.save_model("/data/checkpoints/sft_final")
    tokenizer.save_pretrained("/data/checkpoints/sft_final")
    volume.commit()

    wandb.finish()
    print("Finished")

@app.function(gpu="A100", image=image, volumes={"/data": volume}, timeout=300)
def sanity_check():
    from vllm import LLM, SamplingParams

    llm = LLM(model="/data/checkpoints/sft_final", enforce_eager=True)
    params = SamplingParams(
        temperature=0,
        max_tokens=512,
        stop=["<|im_end|>"],
    )

    SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."
    problems = [
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest for $2 per egg. How much does she make daily?",
        "A store had 150 items. They sold 30% on Monday and 25% of the remaining on Tuesday. How many items are left?",
        "Tom reads 40 pages per hour. He reads for 2 hours a day for 5 days. How many pages did he read in total?",
    ]

    prompts = [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{p}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for p in problems
    ]

    outputs = llm.generate(prompts, params)
    for problem, output in zip(problems, outputs):
        print(f"Problem:  {problem}")
        print(f"Output:   {output.outputs[0].text}")
        print()

@app.function(gpu="A100", image=image, volumes={"/data": volume}, timeout=1200)
def evaluate_sft():
    import re
    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    llm = LLM(model="/data/checkpoints/sft_final", enforce_eager=True)
    params = SamplingParams(temperature=0, max_tokens=512, stop=["<|im_end|>"])

    test_ds = load_dataset("openai/gsm8k", "main")["test"]
    SYSTEM  = "Please reason step by step, and put your final answer within \\boxed{}."

    prompts = [
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{example['question']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for example in test_ds
    ]

    outputs = llm.generate(prompts, params)

    correct = 0
    for example, output in zip(test_ds, outputs):
        response = output.outputs[0].text
        match = re.search(r"\\boxed\{([^}]+)\}", response)
        predicted = match.group(1).strip() if match else ""
        ground_truth = example["answer"].split("####")[1].strip()

        if predicted == ground_truth:
            correct += 1

    accuracy = correct / len(test_ds)
    print(f"GSM8K test accuracy: {accuracy:.2f} ({correct}/{len(test_ds)})")
    return accuracy

@app.local_entrypoint()
def main():
    evaluate_sft.remote()
