from datasets import load_from_disk

sft_ds = load_from_disk("./data/sft_train")

print(sft_ds[0])